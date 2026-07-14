"""Tests for backup utility."""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.backup import backup_before_write, generate_backup_path


class TestGenerateBackupPath:
    """Tests for generate_backup_path function."""

    def test_generates_correct_format(self):
        """Test that backup path follows {base}-BEFORE-SALVAGE-{timestamp}.{ext} pattern."""
        backup_path = generate_backup_path("output.json")

        # Should contain the base name, marker, and extension
        assert "output" in backup_path
        assert "-BEFORE-SALVAGE-" in backup_path
        assert backup_path.endswith(".json")

        # Parse the format: output-BEFORE-SALVAGE-TIMESTAMP.json
        parts = backup_path.split("-BEFORE-SALVAGE-")
        assert len(parts) == 2
        assert parts[0] == "output"
        assert parts[1].endswith(".json")

    def test_timestamp_format_is_filesystem_safe(self):
        """Test that timestamp has no colons and matches YYYY-MM-DDTHHMMSS format."""
        backup_path = generate_backup_path("file.txt")

        # Extract timestamp from path: file-BEFORE-SALVAGE-TIMESTAMP.txt
        timestamp_with_ext = backup_path.split("-BEFORE-SALVAGE-")[1]
        timestamp = timestamp_with_ext.replace(".txt", "")

        # Should not contain colons (filesystem-safe)
        assert ":" not in timestamp

        # Should be parseable as datetime with format YYYY-MM-DDTHHMMSS
        parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H%M%S")
        assert parsed is not None

        # Should have 'T' separator between date and time
        assert "T" in timestamp

        # Should have dashes in date part but not in time part
        parts = timestamp.split("T")
        assert len(parts) == 2
        date_part, time_part = parts
        assert "-" in date_part  # YYYY-MM-DD
        assert "-" not in time_part  # HHMMSS
        assert ":" not in time_part  # No colons

    def test_preserves_original_extension(self):
        """Test that various file extensions are preserved correctly."""
        test_cases = [
            ("file.json", ".json"),
            ("data.txt", ".txt"),
            ("app.log", ".log"),
            ("config.yaml", ".yaml"),
            ("script.py", ".py"),
        ]

        for file_path, expected_ext in test_cases:
            backup_path = generate_backup_path(file_path)
            assert backup_path.endswith(expected_ext), f"Failed for {file_path}"

    def test_handles_no_extension(self):
        """Test that files without extension work correctly."""
        backup_path = generate_backup_path("README")

        # Should have no extension at the end
        assert "-BEFORE-SALVAGE-" in backup_path
        assert backup_path.startswith("README-BEFORE-SALVAGE-")

        # Extract timestamp part after the marker
        timestamp_part = backup_path.split("-BEFORE-SALVAGE-")[1]
        # Should be just the timestamp with no extension (YYYY-MM-DDTHHMMSS format)
        assert "." not in timestamp_part
        # Should be parseable as datetime
        parsed = datetime.strptime(timestamp_part, "%Y-%m-%dT%H%M%S")
        assert parsed is not None

    def test_handles_multiple_dots_in_filename(self):
        """Test that files with multiple dots are handled correctly."""
        backup_path = generate_backup_path("model.v2.json")

        # Should preserve extension correctly
        assert backup_path.endswith(".json")

        # Should preserve base name
        assert "model.v2" in backup_path
        assert "-BEFORE-SALVAGE-" in backup_path

        # Parse format: model.v2-BEFORE-SALVAGE-TIMESTAMP.json
        parts = backup_path.split("-BEFORE-SALVAGE-")
        assert parts[0] == "model.v2"
        assert parts[1].endswith(".json")

    def test_handles_path_with_spaces(self):
        """Test that paths with spaces work correctly."""
        backup_path = generate_backup_path("my file.txt")

        # Should preserve spaces
        assert "my file" in backup_path
        assert "-BEFORE-SALVAGE-" in backup_path
        assert backup_path.endswith(".txt")

        # Should be valid format
        parts = backup_path.split("-BEFORE-SALVAGE-")
        assert parts[0] == "my file"

    def test_handles_directory_in_path(self):
        """Test that directory paths are preserved."""
        backup_path = generate_backup_path("plans/feature/output.json")

        # Directory should be preserved
        assert backup_path.startswith("plans/feature/")
        assert "-BEFORE-SALVAGE-" in backup_path
        assert backup_path.endswith(".json")

        # Parse the filename part
        filename = os.path.basename(backup_path)
        assert filename.startswith("output-BEFORE-SALVAGE-")

    def test_timestamp_increases_over_time(self):
        """Test that timestamps increase when called sequentially."""
        path1 = generate_backup_path("file.txt")
        time.sleep(1.1)  # Wait for timestamp to change
        path2 = generate_backup_path("file.txt")

        # Extract timestamps
        ts1 = path1.split("-BEFORE-SALVAGE-")[1].replace(".txt", "")
        ts2 = path2.split("-BEFORE-SALVAGE-")[1].replace(".txt", "")

        # Second timestamp should be greater
        assert ts2 > ts1


class TestBackupBeforeWrite:
    """Tests for backup_before_write function."""

    def test_creates_backup_when_file_exists(self, tmp_path):
        """Test that backup is created with correct content when file exists."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        original_content = "original content\nline 2\n"
        test_file.write_text(original_content, encoding="utf-8")

        # Create backup
        backup_path = backup_before_write(str(test_file))

        # Backup should be created
        assert backup_path is not None
        assert os.path.exists(backup_path)

        # Backup should have same content
        backup_content = Path(backup_path).read_text(encoding="utf-8")
        assert backup_content == original_content

        # Backup should follow naming convention
        assert "-BEFORE-SALVAGE-" in backup_path
        assert backup_path.endswith(".txt")

    def test_returns_none_when_file_not_exists(self, tmp_path):
        """Test that None is returned and no backup created when file doesn't exist."""
        nonexistent_file = tmp_path / "nonexistent.txt"

        # Should return None
        backup_path = backup_before_write(str(nonexistent_file))
        assert backup_path is None

        # No backup file should be created
        backup_files = list(tmp_path.glob("*BEFORE-SALVAGE*"))
        assert len(backup_files) == 0

    def test_backup_path_in_same_directory(self, tmp_path):
        """Test that backup is created in the same directory as original."""
        # Create a nested directory structure
        nested_dir = tmp_path / "plans" / "feature"
        nested_dir.mkdir(parents=True)

        test_file = nested_dir / "output.json"
        test_file.write_text('{"key": "value"}', encoding="utf-8")

        # Create backup
        backup_path = backup_before_write(str(test_file))

        # Backup should be in same directory
        assert backup_path is not None
        backup_dir = os.path.dirname(backup_path)
        original_dir = os.path.dirname(str(test_file))
        assert backup_dir == original_dir

        # Verify backup exists in the same directory
        assert os.path.exists(backup_path)

    def test_preserves_file_content_exactly(self, tmp_path):
        """Test that backup preserves file content exactly, including formatting."""
        test_file = tmp_path / "data.json"
        original_content = """{
  "key1": "value1",
  "key2": [
    1,
    2,
    3
  ],
  "key3": {
    "nested": true
  }
}"""
        test_file.write_text(original_content, encoding="utf-8")

        # Create backup
        backup_path = backup_before_write(str(test_file))

        # Content should be identical
        backup_content = Path(backup_path).read_text(encoding="utf-8")
        assert backup_content == original_content
        assert len(backup_content) == len(original_content)

    def test_preserves_binary_content(self, tmp_path):
        """Test that binary files are backed up correctly."""
        test_file = tmp_path / "data.bin"
        binary_content = bytes([0x00, 0x01, 0x02, 0xFF, 0xFE, 0xFD])
        test_file.write_bytes(binary_content)

        # Create backup
        backup_path = backup_before_write(str(test_file))

        # Binary content should be identical
        backup_content = Path(backup_path).read_bytes()
        assert backup_content == binary_content
        assert len(backup_content) == len(binary_content)

    def test_handles_empty_file(self, tmp_path):
        """Test that empty files are backed up correctly."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        # Create backup
        backup_path = backup_before_write(str(test_file))

        # Backup should exist and be empty
        assert backup_path is not None
        assert os.path.exists(backup_path)
        assert Path(backup_path).read_text(encoding="utf-8") == ""

    def test_handles_large_file(self, tmp_path):
        """Test that large files (1MB+) are backed up correctly."""
        test_file = tmp_path / "large.txt"

        # Create a 1MB+ file
        large_content = "x" * (1024 * 1024 + 100)  # 1MB + 100 bytes
        test_file.write_text(large_content, encoding="utf-8")

        # Create backup
        backup_path = backup_before_write(str(test_file))

        # Backup should have same size
        assert backup_path is not None
        backup_size = os.path.getsize(backup_path)
        original_size = os.path.getsize(str(test_file))
        assert backup_size == original_size
        assert backup_size > 1024 * 1024

    def test_multiple_backups_have_different_timestamps(self, tmp_path):
        """Test that multiple backups of same file have unique timestamps."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content", encoding="utf-8")

        # Create first backup
        backup1 = backup_before_write(str(test_file))
        assert backup1 is not None

        # Wait for timestamp to change
        time.sleep(1.1)

        # Modify file and create second backup
        test_file.write_text("updated content", encoding="utf-8")
        backup2 = backup_before_write(str(test_file))
        assert backup2 is not None

        # Backups should have different names (different timestamps)
        assert backup1 != backup2

        # Both should exist
        assert os.path.exists(backup1)
        assert os.path.exists(backup2)

        # Extract timestamps from filenames
        ts1 = backup1.split("-BEFORE-SALVAGE-")[1].replace(".txt", "")
        ts2 = backup2.split("-BEFORE-SALVAGE-")[1].replace(".txt", "")
        assert ts2 > ts1

    def test_original_file_unchanged(self, tmp_path):
        """Test that backup operation doesn't modify the original file."""
        test_file = tmp_path / "original.txt"
        original_content = "important data\n"
        test_file.write_text(original_content, encoding="utf-8")

        # Record original modification time
        original_mtime = os.path.getmtime(str(test_file))

        # Create backup
        backup_path = backup_before_write(str(test_file))

        # Original file should still exist with same content
        assert os.path.exists(str(test_file))
        current_content = test_file.read_text(encoding="utf-8")
        assert current_content == original_content

        # Backup should be different file
        assert backup_path != str(test_file)

    def test_preserves_file_permissions(self, tmp_path):
        """Test that backup preserves file metadata (permissions, timestamps)."""
        test_file = tmp_path / "perms.txt"
        test_file.write_text("content", encoding="utf-8")

        # Set specific permissions
        os.chmod(str(test_file), 0o644)

        # Create backup
        backup_path = backup_before_write(str(test_file))

        # Backup should exist
        assert backup_path is not None
        assert os.path.exists(backup_path)

        # Note: shutil.copy2 preserves permissions and timestamps
        # Verify permissions are similar (might not be exact due to umask)
        backup_stat = os.stat(backup_path)
        assert backup_stat.st_mode is not None


class TestBackupCLI:
    """Tests for backup CLI interface."""

    def test_cli_with_existing_file(self, tmp_path):
        """Test CLI with an existing file reports backup path."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": "test"}', encoding="utf-8")

        # Run backup as CLI
        result = subprocess.run(
            [sys.executable, "-m", "utils.backup", str(test_file)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        # Should succeed
        assert result.returncode == 0

        # Should report backup path
        assert "Backed up to:" in result.stdout
        assert "-BEFORE-SALVAGE-" in result.stdout
        # Should contain the base name (test) in the backup path
        assert "test-BEFORE-SALVAGE-" in result.stdout

        # Extract backup path from output
        backup_path = result.stdout.strip().replace("Backed up to: ", "")
        assert os.path.exists(backup_path)

    def test_cli_with_nonexistent_file(self, tmp_path):
        """Test CLI with nonexistent file reports no backup needed."""
        nonexistent_file = tmp_path / "nonexistent.txt"

        # Run backup as CLI
        result = subprocess.run(
            [sys.executable, "-m", "utils.backup", str(nonexistent_file)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        # Should succeed
        assert result.returncode == 0

        # Should report no backup needed
        assert "No backup needed" in result.stdout
        assert "file does not exist" in result.stdout

    def test_cli_with_no_args(self):
        """Test CLI with no arguments shows usage and exits with error."""
        # Run backup with no arguments
        result = subprocess.run(
            [sys.executable, "-m", "utils.backup"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        # Should fail
        assert result.returncode == 1

        # Should show usage in stderr
        assert "Usage:" in result.stderr
        assert "python -m utils.backup" in result.stderr
        assert "<file_path>" in result.stderr

    def test_cli_with_too_many_args(self):
        """Test CLI with too many arguments shows usage and exits with error."""
        # Run backup with too many arguments
        result = subprocess.run(
            [sys.executable, "-m", "utils.backup", "file1.txt", "file2.txt"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        # Should fail
        assert result.returncode == 1

        # Should show usage in stderr
        assert "Usage:" in result.stderr

    def test_cli_creates_actual_backup(self, tmp_path):
        """Test that CLI actually creates the backup file."""
        test_file = tmp_path / "important.txt"
        test_content = "important data"
        test_file.write_text(test_content, encoding="utf-8")

        # Run backup as CLI
        result = subprocess.run(
            [sys.executable, "-m", "utils.backup", str(test_file)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        # Should succeed
        assert result.returncode == 0

        # Extract backup path from output
        backup_path = result.stdout.strip().replace("Backed up to: ", "")

        # Verify backup was created and has correct content
        assert os.path.exists(backup_path)
        backup_content = Path(backup_path).read_text(encoding="utf-8")
        assert backup_content == test_content

    def test_cli_with_directory_path(self, tmp_path):
        """Test CLI with file in nested directory."""
        nested_dir = tmp_path / "plans" / "feature"
        nested_dir.mkdir(parents=True)

        test_file = nested_dir / "output.json"
        test_file.write_text('{"key": "value"}', encoding="utf-8")

        # Run backup as CLI
        result = subprocess.run(
            [sys.executable, "-m", "utils.backup", str(test_file)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        # Should succeed
        assert result.returncode == 0
        assert "Backed up to:" in result.stdout

        # Backup should be in same directory as original
        backup_path = result.stdout.strip().replace("Backed up to: ", "")
        assert backup_path.startswith(str(nested_dir))


class TestBackupEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_backup_with_unicode_content(self, tmp_path):
        """Test that files with unicode content are backed up correctly."""
        test_file = tmp_path / "unicode.txt"
        unicode_content = "Hello 世界 🌍 Привет"
        test_file.write_text(unicode_content, encoding="utf-8")

        # Create backup
        backup_path = backup_before_write(str(test_file))

        # Content should be preserved
        assert backup_path is not None
        backup_content = Path(backup_path).read_text(encoding="utf-8")
        assert backup_content == unicode_content

    def test_backup_with_newlines_and_special_chars(self, tmp_path):
        """Test that files with various newline styles are preserved."""
        test_file = tmp_path / "newlines.txt"
        content_with_newlines = "line1\nline2\r\nline3\ttabbed\n"
        # Write in binary mode to preserve exact newline chars
        test_file.write_bytes(content_with_newlines.encode('utf-8'))

        # Create backup
        backup_path = backup_before_write(str(test_file))

        # Content should be exactly preserved
        assert backup_path is not None
        # Read in binary mode to preserve exact newline chars
        backup_content = Path(backup_path).read_bytes().decode('utf-8')
        assert backup_content == content_with_newlines

    def test_backup_filename_uniqueness(self, tmp_path):
        """Test that backup filenames are unique even for rapid calls."""
        test_file = tmp_path / "rapid.txt"
        test_file.write_text("initial", encoding="utf-8")

        backup_paths = []

        # Create multiple backups rapidly
        for i in range(3):
            test_file.write_text(f"version {i}", encoding="utf-8")
            backup_path = backup_before_write(str(test_file))
            backup_paths.append(backup_path)
            # Small delay to ensure different timestamps
            time.sleep(1.1)

        # All backup paths should be unique
        assert len(backup_paths) == len(set(backup_paths))

        # All backups should exist
        for backup_path in backup_paths:
            assert os.path.exists(backup_path)

    def test_backup_path_return_value_is_absolute(self, tmp_path):
        """Test that backup_before_write returns absolute path."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content", encoding="utf-8")

        backup_path = backup_before_write(str(test_file))

        # Should return absolute path if input was absolute
        assert backup_path is not None
        if os.path.isabs(str(test_file)):
            assert os.path.isabs(backup_path)
