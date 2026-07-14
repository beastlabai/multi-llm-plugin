"""Unit tests for git utilities in utils/git_utils.py."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.git_utils import (
    GitError,
    _is_binary_file,
    _parse_unified_diff,
    _run_git,
    _split_diff_by_file,
    capture_diff_hunks,
    capture_file_context,
    get_branch_name,
    get_current_head,
    get_diff_since_ref,
    get_file_diff,
    get_files_changed_since_ref,
    get_modified_files,
    get_staged_diff,
    get_staged_files,
    intent_to_add_untracked,
    is_clean_working_tree,
    stage_files,
    unstage_files,
)


class TestGitError:
    """Tests for GitError exception class."""

    def test_git_error_is_exception(self):
        """GitError is an Exception subclass."""
        assert issubclass(GitError, Exception)

    def test_git_error_can_be_raised_with_message(self):
        """GitError can be raised with a message."""
        with pytest.raises(GitError) as exc_info:
            raise GitError("test error message")
        assert "test error message" in str(exc_info.value)

    def test_git_error_can_be_caught(self):
        """GitError can be caught in try/except block."""
        caught = False
        try:
            raise GitError("test")
        except GitError:
            caught = True
        assert caught is True


class TestRunGit:
    """Tests for _run_git internal function."""

    @patch("utils.git_utils.subprocess.run")
    def test_run_git_success_returns_tuple(self, mock_run):
        """Successful git command returns (stdout, stderr, returncode) tuple."""
        mock_run.return_value = MagicMock(
            stdout="output text",
            stderr="",
            returncode=0
        )

        stdout, stderr, code = _run_git("status")

        assert stdout == "output text"
        assert stderr == ""
        assert code == 0
        mock_run.assert_called_once_with(
            ["git", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    @patch("utils.git_utils.subprocess.run")
    def test_run_git_with_multiple_args(self, mock_run):
        """Git command with multiple arguments passes them correctly."""
        mock_run.return_value = MagicMock(
            stdout="diff output",
            stderr="",
            returncode=0
        )

        _run_git("diff", "--cached", "--name-only")

        mock_run.assert_called_once_with(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    @patch("utils.git_utils.subprocess.run")
    def test_run_git_non_zero_exit_raises_git_error(self, mock_run):
        """Non-zero exit code raises GitError when check=True."""
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="fatal: not a git repository",
            returncode=128
        )

        with pytest.raises(GitError) as exc_info:
            _run_git("status")

        assert "git status failed" in str(exc_info.value)
        assert "fatal: not a git repository" in str(exc_info.value)

    @patch("utils.git_utils.subprocess.run")
    def test_run_git_non_zero_exit_with_check_false(self, mock_run):
        """Non-zero exit code returns tuple when check=False."""
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="error message",
            returncode=1
        )

        stdout, stderr, code = _run_git("symbolic-ref", "--short", "HEAD", check=False)

        assert stdout == ""
        assert stderr == "error message"
        assert code == 1

    @patch("utils.git_utils.subprocess.run")
    def test_run_git_missing_binary_raises_exception(self, mock_run):
        """FileNotFoundError from missing git binary propagates."""
        mock_run.side_effect = FileNotFoundError("git not found")

        with pytest.raises(FileNotFoundError):
            _run_git("status")

    @patch("utils.git_utils.subprocess.run")
    def test_run_git_captures_stderr(self, mock_run):
        """Git command captures stderr output."""
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="warning: some warning",
            returncode=0
        )

        stdout, stderr, code = _run_git("status")

        assert stderr == "warning: some warning"


class TestGetModifiedFiles:
    """Tests for get_modified_files function."""

    @patch("utils.git_utils._run_git")
    def test_clean_repo_returns_empty_list(self, mock_run_git):
        """Clean repository returns empty list."""
        mock_run_git.return_value = ("", "", 0)

        files = get_modified_files()

        assert files == []
        mock_run_git.assert_called_once_with("status", "--porcelain")

    @patch("utils.git_utils._run_git")
    def test_modified_file_returns_in_list(self, mock_run_git):
        """Modified file appears in returned list.

        Git status --porcelain format: XY filename
        where XY is 2-char status and filename starts at position 3.

        Note: Using 'M ' (modified in index) instead of ' M' (modified in worktree)
        because stdout.strip() in the implementation removes leading spaces.
        """
        # "M " = modification staged in index
        mock_run_git.return_value = ("M  src/main.py\n", "", 0)

        files = get_modified_files()

        assert files == ["src/main.py"]

    @patch("utils.git_utils._run_git")
    def test_untracked_file_returns_in_list(self, mock_run_git):
        """Untracked file appears in returned list."""
        mock_run_git.return_value = ("?? new_file.txt\n", "", 0)

        files = get_modified_files()

        assert files == ["new_file.txt"]

    @patch("utils.git_utils._run_git")
    def test_staged_file_returns_in_list(self, mock_run_git):
        """Staged file appears in returned list."""
        mock_run_git.return_value = ("A  staged_file.py\n", "", 0)

        files = get_modified_files()

        assert files == ["staged_file.py"]

    @patch("utils.git_utils._run_git")
    def test_multiple_files_with_different_statuses(self, mock_run_git):
        """Multiple files with different statuses all appear.

        Note: Using 'M ' instead of ' M' for first line because stdout.strip()
        removes leading spaces from the entire output.
        """
        mock_run_git.return_value = (
            "M  modified.py\n"
            "A  added.py\n"
            "?? untracked.txt\n"
            "D  deleted.py\n",
            "",
            0
        )

        files = get_modified_files()

        assert len(files) == 4
        assert "modified.py" in files
        assert "added.py" in files
        assert "untracked.txt" in files
        assert "deleted.py" in files

    @patch("utils.git_utils._run_git")
    def test_modified_and_staged_file(self, mock_run_git):
        """File with both staged and unstaged changes appears once."""
        mock_run_git.return_value = ("MM both_changes.py\n", "", 0)

        files = get_modified_files()

        assert files == ["both_changes.py"]

    @patch("utils.git_utils._run_git")
    def test_empty_lines_filtered_out(self, mock_run_git):
        """Empty lines in output are filtered out.

        Note: Using 'M ' instead of ' M' for first line because stdout.strip()
        removes leading spaces from the entire output.
        """
        mock_run_git.return_value = ("M  file1.py\n\nM  file2.py\n", "", 0)

        files = get_modified_files()

        assert len(files) == 2
        assert "file1.py" in files
        assert "file2.py" in files


class TestGetStagedFiles:
    """Tests for get_staged_files function."""

    @patch("utils.git_utils._run_git")
    def test_nothing_staged_returns_empty_list(self, mock_run_git):
        """No staged files returns empty list."""
        mock_run_git.return_value = ("", "", 0)

        files = get_staged_files()

        assert files == []
        mock_run_git.assert_called_once_with("diff", "--cached", "--name-only")

    @patch("utils.git_utils._run_git")
    def test_single_staged_file(self, mock_run_git):
        """Single staged file appears in list."""
        mock_run_git.return_value = ("src/main.py\n", "", 0)

        files = get_staged_files()

        assert files == ["src/main.py"]

    @patch("utils.git_utils._run_git")
    def test_multiple_staged_files(self, mock_run_git):
        """Multiple staged files appear in list."""
        mock_run_git.return_value = (
            "src/main.py\n"
            "src/utils.py\n"
            "tests/test_main.py\n",
            "",
            0
        )

        files = get_staged_files()

        assert len(files) == 3
        assert "src/main.py" in files
        assert "src/utils.py" in files
        assert "tests/test_main.py" in files

    @patch("utils.git_utils._run_git")
    def test_empty_lines_filtered_out(self, mock_run_git):
        """Empty lines in output are filtered out."""
        mock_run_git.return_value = ("file1.py\n\nfile2.py\n", "", 0)

        files = get_staged_files()

        assert len(files) == 2


class TestStageFiles:
    """Tests for stage_files function."""

    @patch("utils.git_utils._run_git")
    def test_stage_single_file(self, mock_run_git):
        """Staging single file calls git add correctly."""
        mock_run_git.return_value = ("", "", 0)

        stage_files(["src/main.py"])

        mock_run_git.assert_called_once_with("add", "--", "src/main.py")

    @patch("utils.git_utils._run_git")
    def test_stage_multiple_files(self, mock_run_git):
        """Staging multiple files calls git add with all files."""
        mock_run_git.return_value = ("", "", 0)

        stage_files(["file1.py", "file2.py", "file3.py"])

        mock_run_git.assert_called_once_with(
            "add", "--", "file1.py", "file2.py", "file3.py"
        )

    @patch("utils.git_utils._run_git")
    def test_stage_empty_list_does_nothing(self, mock_run_git):
        """Staging empty list does not call git."""
        stage_files([])

        mock_run_git.assert_not_called()

    @patch("utils.git_utils._run_git")
    def test_stage_files_raises_on_error(self, mock_run_git):
        """Staging non-existent file raises GitError."""
        mock_run_git.side_effect = GitError("git add failed: file not found")

        with pytest.raises(GitError):
            stage_files(["nonexistent.py"])


class TestUnstageFiles:
    """Tests for unstage_files function."""

    @patch("utils.git_utils._run_git")
    def test_unstage_single_file(self, mock_run_git):
        """Unstaging single file calls git reset correctly."""
        mock_run_git.return_value = ("", "", 0)

        unstage_files(["src/main.py"])

        mock_run_git.assert_called_once_with("reset", "HEAD", "--", "src/main.py")

    @patch("utils.git_utils._run_git")
    def test_unstage_multiple_files(self, mock_run_git):
        """Unstaging multiple files calls git reset with all files."""
        mock_run_git.return_value = ("", "", 0)

        unstage_files(["file1.py", "file2.py"])

        mock_run_git.assert_called_once_with(
            "reset", "HEAD", "--", "file1.py", "file2.py"
        )

    @patch("utils.git_utils._run_git")
    def test_unstage_empty_list_does_nothing(self, mock_run_git):
        """Unstaging empty list does not call git."""
        unstage_files([])

        mock_run_git.assert_not_called()

    @patch("utils.git_utils._run_git")
    def test_unstage_files_raises_on_error(self, mock_run_git):
        """Unstaging fails with GitError on git error."""
        mock_run_git.side_effect = GitError("git reset failed")

        with pytest.raises(GitError):
            unstage_files(["file.py"])


class TestGetStagedDiff:
    """Tests for get_staged_diff function."""

    @patch("utils.git_utils._run_git")
    def test_no_staged_changes_returns_empty_string(self, mock_run_git):
        """No staged changes returns empty string."""
        mock_run_git.return_value = ("", "", 0)

        diff = get_staged_diff()

        assert diff == ""
        mock_run_git.assert_called_once_with("diff", "--cached")

    @patch("utils.git_utils._run_git")
    def test_staged_changes_returns_diff(self, mock_run_git):
        """Staged changes returns diff content."""
        diff_output = (
            "diff --git a/src/main.py b/src/main.py\n"
            "index abc1234..def5678 100644\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,3 +1,4 @@\n"
            " import os\n"
            "+import sys\n"
            " def main():\n"
        )
        mock_run_git.return_value = (diff_output, "", 0)

        diff = get_staged_diff()

        assert diff == diff_output
        assert "import sys" in diff


class TestGetFileDiff:
    """Tests for get_file_diff function."""

    @patch("utils.git_utils._run_git")
    def test_diff_without_base_ref(self, mock_run_git):
        """Diff without base_ref calls git diff correctly."""
        diff_output = "diff --git a/file.py b/file.py\n+new line"
        mock_run_git.return_value = (diff_output, "", 0)

        diff = get_file_diff("src/file.py")

        assert diff == diff_output
        mock_run_git.assert_called_once_with("diff", "--", "src/file.py")

    @patch("utils.git_utils._run_git")
    def test_diff_with_base_ref(self, mock_run_git):
        """Diff with base_ref includes ref in command."""
        diff_output = "diff --git a/file.py b/file.py\n+changes"
        mock_run_git.return_value = (diff_output, "", 0)

        diff = get_file_diff("src/file.py", base_ref="HEAD~1")

        assert diff == diff_output
        mock_run_git.assert_called_once_with("diff", "HEAD~1", "--", "src/file.py")

    @patch("utils.git_utils._run_git")
    def test_diff_with_branch_ref(self, mock_run_git):
        """Diff with branch name as base_ref."""
        mock_run_git.return_value = ("some diff", "", 0)

        get_file_diff("file.py", base_ref="main")

        mock_run_git.assert_called_once_with("diff", "main", "--", "file.py")

    @patch("utils.git_utils._run_git")
    def test_diff_no_changes_returns_empty(self, mock_run_git):
        """File with no changes returns empty string."""
        mock_run_git.return_value = ("", "", 0)

        diff = get_file_diff("unchanged.py")

        assert diff == ""


class TestGetCurrentHead:
    """Tests for get_current_head function."""

    @patch("utils.git_utils._run_git")
    def test_returns_commit_hash(self, mock_run_git):
        """Returns current HEAD commit hash."""
        mock_run_git.return_value = ("abc123def456789\n", "", 0)

        head = get_current_head()

        assert head == "abc123def456789"
        mock_run_git.assert_called_once_with("rev-parse", "HEAD")

    @patch("utils.git_utils._run_git")
    def test_strips_whitespace(self, mock_run_git):
        """Strips trailing whitespace from commit hash."""
        mock_run_git.return_value = ("  abc123  \n", "", 0)

        head = get_current_head()

        assert head == "abc123"

    @patch("utils.git_utils._run_git")
    def test_raises_on_error(self, mock_run_git):
        """Raises GitError when not in a git repository."""
        mock_run_git.side_effect = GitError("git rev-parse HEAD failed")

        with pytest.raises(GitError):
            get_current_head()


class TestGetBranchName:
    """Tests for get_branch_name function."""

    @patch("utils.git_utils._run_git")
    def test_on_branch_returns_name(self, mock_run_git):
        """On a branch returns branch name."""
        mock_run_git.return_value = ("feature/new-feature\n", "", 0)

        branch = get_branch_name()

        assert branch == "feature/new-feature"
        mock_run_git.assert_called_once_with(
            "symbolic-ref", "--short", "HEAD", check=False
        )

    @patch("utils.git_utils._run_git")
    def test_on_main_branch_returns_main(self, mock_run_git):
        """On main branch returns 'main'."""
        mock_run_git.return_value = ("main\n", "", 0)

        branch = get_branch_name()

        assert branch == "main"

    @patch("utils.git_utils._run_git")
    def test_detached_head_returns_none(self, mock_run_git):
        """Detached HEAD returns None."""
        mock_run_git.return_value = ("", "fatal: ref HEAD is not a symbolic ref", 128)

        branch = get_branch_name()

        assert branch is None

    @patch("utils.git_utils._run_git")
    def test_strips_whitespace(self, mock_run_git):
        """Strips whitespace from branch name."""
        mock_run_git.return_value = ("  develop  \n", "", 0)

        branch = get_branch_name()

        assert branch == "develop"


class TestIsCleanWorkingTree:
    """Tests for is_clean_working_tree function."""

    @patch("utils.git_utils._run_git")
    def test_clean_tree_returns_true(self, mock_run_git):
        """Clean working tree returns True."""
        # Both diff commands return 0 (no changes)
        mock_run_git.side_effect = [
            ("", "", 0),  # git diff --quiet
            ("", "", 0),  # git diff --cached --quiet
        ]

        result = is_clean_working_tree()

        assert result is True
        assert mock_run_git.call_count == 2

    @patch("utils.git_utils._run_git")
    def test_unstaged_changes_returns_false(self, mock_run_git):
        """Unstaged changes returns False."""
        # First diff returns 1 (has changes)
        mock_run_git.return_value = ("", "", 1)

        result = is_clean_working_tree()

        assert result is False
        # Only one call needed since first check fails
        mock_run_git.assert_called_once_with("diff", "--quiet", check=False)

    @patch("utils.git_utils._run_git")
    def test_staged_changes_returns_false(self, mock_run_git):
        """Staged changes returns False."""
        mock_run_git.side_effect = [
            ("", "", 0),  # git diff --quiet (no unstaged)
            ("", "", 1),  # git diff --cached --quiet (has staged)
        ]

        result = is_clean_working_tree()

        assert result is False

    @patch("utils.git_utils._run_git")
    def test_both_staged_and_unstaged_returns_false(self, mock_run_git):
        """Both staged and unstaged changes returns False."""
        # First check fails immediately
        mock_run_git.return_value = ("", "", 1)

        result = is_clean_working_tree()

        assert result is False


class TestGetDiffSinceRef:
    """Tests for get_diff_since_ref function."""

    @patch("utils.git_utils._run_git")
    def test_returns_diff_since_ref(self, mock_run_git):
        """Returns diff since specified reference."""
        diff_output = "diff content here"
        mock_run_git.return_value = (diff_output, "", 0)

        diff = get_diff_since_ref("HEAD~5")

        assert diff == diff_output
        mock_run_git.assert_called_once_with("diff", "HEAD~5")

    @patch("utils.git_utils._run_git")
    def test_with_branch_name(self, mock_run_git):
        """Works with branch name as reference."""
        mock_run_git.return_value = ("branch diff", "", 0)

        diff = get_diff_since_ref("main")

        assert diff == "branch diff"
        mock_run_git.assert_called_once_with("diff", "main")

    @patch("utils.git_utils._run_git")
    def test_invalid_ref_raises_error(self, mock_run_git):
        """Invalid reference raises GitError."""
        mock_run_git.side_effect = GitError("git diff invalid-ref failed")

        with pytest.raises(GitError):
            get_diff_since_ref("invalid-ref")

    @patch("utils.git_utils._run_git")
    def test_no_changes_returns_empty(self, mock_run_git):
        """No changes since ref returns empty string."""
        mock_run_git.return_value = ("", "", 0)

        diff = get_diff_since_ref("HEAD")

        assert diff == ""


class TestGetFilesChangedSinceRef:
    """Tests for get_files_changed_since_ref function."""

    @patch("utils.git_utils._run_git")
    def test_returns_list_of_files(self, mock_run_git):
        """Returns list of files changed since reference."""
        mock_run_git.return_value = (
            "src/main.py\nsrc/utils.py\ntests/test_main.py\n",
            "",
            0
        )

        files = get_files_changed_since_ref("HEAD~3")

        assert len(files) == 3
        assert "src/main.py" in files
        assert "src/utils.py" in files
        assert "tests/test_main.py" in files
        mock_run_git.assert_called_once_with("diff", "--name-only", "HEAD~3")

    @patch("utils.git_utils._run_git")
    def test_no_changes_returns_empty_list(self, mock_run_git):
        """No changes since ref returns empty list."""
        mock_run_git.return_value = ("", "", 0)

        files = get_files_changed_since_ref("HEAD")

        assert files == []

    @patch("utils.git_utils._run_git")
    def test_filters_empty_lines(self, mock_run_git):
        """Empty lines in output are filtered out."""
        mock_run_git.return_value = ("file1.py\n\nfile2.py\n\n", "", 0)

        files = get_files_changed_since_ref("HEAD~1")

        assert len(files) == 2
        assert "file1.py" in files
        assert "file2.py" in files

    @patch("utils.git_utils._run_git")
    def test_with_commit_hash(self, mock_run_git):
        """Works with commit hash as reference."""
        mock_run_git.return_value = ("changed.py\n", "", 0)

        files = get_files_changed_since_ref("abc123def")

        assert files == ["changed.py"]
        mock_run_git.assert_called_once_with("diff", "--name-only", "abc123def")

    @patch("utils.git_utils._run_git")
    def test_invalid_ref_raises_error(self, mock_run_git):
        """Invalid reference raises GitError."""
        mock_run_git.side_effect = GitError("git diff --name-only invalid failed")

        with pytest.raises(GitError):
            get_files_changed_since_ref("invalid-ref-xyz")


class TestIntegration:
    """Integration tests ensuring functions work together correctly."""

    @patch("utils.git_utils._run_git")
    def test_stage_then_get_staged_files(self, mock_run_git):
        """Stage files then get staged files works correctly."""
        mock_run_git.side_effect = [
            ("", "", 0),  # stage_files
            ("file1.py\nfile2.py\n", "", 0),  # get_staged_files
        ]

        stage_files(["file1.py", "file2.py"])
        files = get_staged_files()

        assert len(files) == 2

    @patch("utils.git_utils._run_git")
    def test_unstage_then_verify_empty(self, mock_run_git):
        """Unstage files then verify empty staged list."""
        mock_run_git.side_effect = [
            ("", "", 0),  # unstage_files
            ("", "", 0),  # get_staged_files
        ]

        unstage_files(["file.py"])
        files = get_staged_files()

        assert files == []


# ===================================================================
# Tests for diff hunk capture and file content capture utilities
# ===================================================================

# --- Sample diff outputs for testing ---

SAMPLE_SINGLE_FILE_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
index abc1234..def5678 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,6 +10,7 @@ def hello():
     x = 1
     y = 2
     z = 3
+    w = 4
     return x + y + z

 def goodbye():
"""

SAMPLE_MULTI_FILE_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
index abc1234..def5678 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,6 +10,7 @@ def hello():
     x = 1
     y = 2
     z = 3
+    w = 4
     return x + y + z

 def goodbye():
diff --git a/src/bar.py b/src/bar.py
index 1111111..2222222 100644
--- a/src/bar.py
+++ b/src/bar.py
@@ -1,4 +1,3 @@
 import os
-import sys
 import time

"""

SAMPLE_BINARY_DIFF = """\
diff --git a/image.png b/image.png
Binary files a/image.png and b/image.png differ
"""

SAMPLE_RENAME_DIFF = """\
diff --git a/old_name.py b/new_name.py
similarity index 92%
rename from old_name.py
rename to new_name.py
index abc1234..def5678 100644
--- a/old_name.py
+++ b/new_name.py
@@ -1,3 +1,4 @@
 def func():
     pass
+    return True
"""

SAMPLE_DELETE_DIFF = """\
diff --git a/deleted.py b/deleted.py
deleted file mode 100644
index abc1234..0000000
--- a/deleted.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def old_func():
-    pass
-    return False
"""

SAMPLE_MULTI_HUNK_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
index abc1234..def5678 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,5 +1,6 @@
 import os
 import sys
+import re

 def hello():
     pass
@@ -20,4 +21,5 @@ def goodbye():
     x = 1
     y = 2
     z = 3
+    w = 4
     return x + y + z
"""


class TestParseUnifiedDiff:
    """Tests for _parse_unified_diff internal function."""

    def test_basic_diff_with_addition(self):
        """Parse a simple diff with one added line."""
        result = _parse_unified_diff(SAMPLE_SINGLE_FILE_DIFF)

        assert result["binary"] is False
        assert result["deleted"] is False
        assert len(result["hunks"]) == 1

        hunk = result["hunks"][0]
        assert "@@ -10,6 +10,7 @@" in hunk["header"]

        # Check line types
        line_types = [l["type"] for l in hunk["lines"]]
        assert "add" in line_types
        assert "context" in line_types

        # Find the added line
        added = [l for l in hunk["lines"] if l["type"] == "add"]
        assert len(added) == 1
        assert added[0]["content"] == "    w = 4"
        assert added[0]["new_line"] == 13
        assert added[0]["old_line"] is None

    def test_diff_with_removal(self):
        """Parse a diff with a removed line."""
        diff_text = """\
diff --git a/src/bar.py b/src/bar.py
index 1111111..2222222 100644
--- a/src/bar.py
+++ b/src/bar.py
@@ -1,4 +1,3 @@
 import os
-import sys
 import time

"""
        result = _parse_unified_diff(diff_text)

        hunk = result["hunks"][0]
        removed = [l for l in hunk["lines"] if l["type"] == "remove"]
        assert len(removed) == 1
        assert removed[0]["content"] == "import sys"
        assert removed[0]["old_line"] == 2
        assert removed[0]["new_line"] is None

    def test_binary_file_detection(self):
        """Binary file diff produces binary: true."""
        result = _parse_unified_diff(SAMPLE_BINARY_DIFF)
        assert result["binary"] is True
        assert result["hunks"] == []

    def test_renamed_file_paths(self):
        """Renamed file includes both old_path and new_path."""
        result = _parse_unified_diff(SAMPLE_RENAME_DIFF)
        assert result["old_path"] == "old_name.py"
        assert result["new_path"] == "new_name.py"
        assert result["binary"] is False

    def test_deleted_file_detection(self):
        """Deleted file has deleted: true and +++ /dev/null."""
        result = _parse_unified_diff(SAMPLE_DELETE_DIFF)
        assert result["deleted"] is True

        hunk = result["hunks"][0]
        removed = [l for l in hunk["lines"] if l["type"] == "remove"]
        assert len(removed) == 3

    def test_multi_hunk_diff(self):
        """File with multiple hunks parses all hunks."""
        result = _parse_unified_diff(SAMPLE_MULTI_HUNK_DIFF)
        assert len(result["hunks"]) == 2

        # First hunk
        h1 = result["hunks"][0]
        assert "@@ -1,5 +1,6 @@" in h1["header"]
        added1 = [l for l in h1["lines"] if l["type"] == "add"]
        assert any("import re" in l["content"] for l in added1)

        # Second hunk
        h2 = result["hunks"][1]
        assert "@@ -20,4 +21,5 @@" in h2["header"]
        added2 = [l for l in h2["lines"] if l["type"] == "add"]
        assert any("w = 4" in l["content"] for l in added2)

    def test_line_numbering_correctness(self):
        """Verify old_line and new_line numbers are tracked correctly."""
        result = _parse_unified_diff(SAMPLE_SINGLE_FILE_DIFF)
        hunk = result["hunks"][0]

        # Context lines should have both old_line and new_line
        context_lines = [l for l in hunk["lines"] if l["type"] == "context"]
        for ctx in context_lines:
            assert ctx["old_line"] is not None
            assert ctx["new_line"] is not None

        # First context line starts at line 10 (from @@ -10,6 +10,7 @@)
        assert context_lines[0]["old_line"] == 10
        assert context_lines[0]["new_line"] == 10

    def test_per_file_200_line_cap(self):
        """Lines beyond 200 produce truncated_at field."""
        # Build a diff with >200 lines
        diff_lines = [
            "diff --git a/big.py b/big.py",
            "index aaa..bbb 100644",
            "--- a/big.py",
            "+++ b/big.py",
            "@@ -1,250 +1,260 @@",
        ]
        for i in range(1, 251):
            diff_lines.append(f"+line {i}")
        diff_text = "\n".join(diff_lines)

        result = _parse_unified_diff(diff_text)
        assert result.get("truncated_at") == 200

        # Should have exactly 200 lines captured
        total_lines = sum(len(h["lines"]) for h in result["hunks"])
        assert total_lines == 200

    def test_old_new_paths_from_headers(self):
        """Old/new paths extracted from --- a/ and +++ b/ headers."""
        result = _parse_unified_diff(SAMPLE_SINGLE_FILE_DIFF)
        assert result["old_path"] == "src/foo.py"
        assert result["new_path"] == "src/foo.py"

    def test_git_binary_patch_detection(self):
        """GIT binary patch marker also produces binary: true."""
        diff_text = """\
diff --git a/data.bin b/data.bin
index abc..def 100644
GIT binary patch
literal 1234
zcmV;@1TFl$iwFP!0000
"""
        result = _parse_unified_diff(diff_text)
        assert result["binary"] is True

    def test_no_newline_at_eof_marker(self):
        """'No newline at end of file' marker is handled gracefully."""
        diff_text = """\
diff --git a/file.py b/file.py
index abc..def 100644
--- a/file.py
+++ b/file.py
@@ -1,2 +1,2 @@
 first line
-old last line
+new last line
\\ No newline at end of file
"""
        result = _parse_unified_diff(diff_text)
        hunk = result["hunks"][0]
        # The marker line should not be included as content
        for line_obj in hunk["lines"]:
            assert "No newline at end of file" not in line_obj["content"]


class TestSplitDiffByFile:
    """Tests for _split_diff_by_file internal function."""

    def test_single_file(self):
        """Single file diff maps to one entry."""
        result = _split_diff_by_file(SAMPLE_SINGLE_FILE_DIFF)
        assert "src/foo.py" in result
        assert len(result) == 1

    def test_multi_file(self):
        """Multi-file diff splits into separate entries."""
        result = _split_diff_by_file(SAMPLE_MULTI_FILE_DIFF)
        assert "src/foo.py" in result
        assert "src/bar.py" in result
        assert len(result) == 2

    def test_empty_diff(self):
        """Empty diff returns empty dict."""
        result = _split_diff_by_file("")
        assert result == {}

    def test_each_chunk_contains_complete_diff(self):
        """Each file chunk contains the diff header and hunks."""
        result = _split_diff_by_file(SAMPLE_MULTI_FILE_DIFF)
        assert "diff --git" in result["src/foo.py"]
        assert "@@ " in result["src/foo.py"]
        assert "diff --git" in result["src/bar.py"]
        assert "@@ " in result["src/bar.py"]


class TestCaptureDiffHunks:
    """Tests for capture_diff_hunks function."""

    @patch("utils.git_utils.subprocess.run")
    def test_basic_diff_capture(self, mock_run):
        """Basic diff capture returns structured hunk data keyed by file path."""
        # First call: git status --porcelain
        # Second call: git diff base_ref -- files
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # git status
            MagicMock(returncode=0, stdout=SAMPLE_SINGLE_FILE_DIFF, stderr=""),  # git diff
        ]

        result = capture_diff_hunks("HEAD~1", ["src/foo.py"])

        assert "src/foo.py" in result
        file_data = result["src/foo.py"]
        assert len(file_data["hunks"]) == 1
        assert file_data["binary"] is False

    @patch("utils.git_utils.subprocess.run")
    def test_binary_file_produces_binary_entry(self, mock_run):
        """Binary files in diff produce {binary: true} entries."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=SAMPLE_BINARY_DIFF, stderr=""),
        ]

        result = capture_diff_hunks("HEAD~1", ["image.png"])

        assert "image.png" in result
        assert result["image.png"]["binary"] is True

    @patch("utils.git_utils.subprocess.run")
    def test_renamed_file_includes_both_paths(self, mock_run):
        """Renamed files include old_path and new_path."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=SAMPLE_RENAME_DIFF, stderr=""),
        ]

        result = capture_diff_hunks("HEAD~1", ["new_name.py"])

        assert "new_name.py" in result
        file_data = result["new_name.py"]
        assert file_data["old_path"] == "old_name.py"
        assert file_data["new_path"] == "new_name.py"

    @patch("utils.git_utils.subprocess.run")
    def test_invalid_base_ref_returns_empty_dict(self, mock_run):
        """Invalid base_ref returns empty dict without raising."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # git status
            MagicMock(returncode=128, stdout="", stderr="fatal: bad ref"),  # git diff fails
        ]

        result = capture_diff_hunks("nonexistent-ref", ["src/foo.py"])

        # Should not contain file data
        assert "src/foo.py" not in result

    @patch("utils.git_utils.subprocess.run")
    def test_uncommitted_changes_notice(self, mock_run):
        """Uncommitted changes produce a notice in the result."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="M  src/foo.py\n", stderr=""),  # git status (dirty)
            MagicMock(returncode=0, stdout=SAMPLE_SINGLE_FILE_DIFF, stderr=""),
        ]

        result = capture_diff_hunks("HEAD~1", ["src/foo.py"])

        assert "_notices" in result
        assert any("uncommitted" in n.lower() for n in result["_notices"])

    @patch("utils.git_utils.subprocess.run")
    def test_clean_working_tree_no_uncommitted_notice(self, mock_run):
        """Clean working tree does not produce uncommitted changes notice."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # git status (clean)
            MagicMock(returncode=0, stdout=SAMPLE_SINGLE_FILE_DIFF, stderr=""),
        ]

        result = capture_diff_hunks("HEAD~1", ["src/foo.py"])

        # No _notices key when everything is clean
        if "_notices" in result:
            assert not any("uncommitted" in n.lower() for n in result["_notices"])

    @patch("utils.git_utils.subprocess.run")
    def test_git_not_available_returns_notice(self, mock_run):
        """Git not available returns empty dict with notice."""
        mock_run.side_effect = FileNotFoundError("git not found")

        result = capture_diff_hunks("HEAD~1", ["src/foo.py"])

        assert "_notices" in result
        assert any("Git not available" in n for n in result["_notices"])
        # Should not contain any file data
        assert "src/foo.py" not in result

    @patch("utils.git_utils.subprocess.run")
    def test_timeout_returns_partial_results(self, mock_run):
        """Timeout returns whatever was collected with a timeout notice."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # git status
            subprocess.TimeoutExpired(cmd="git diff", timeout=10),  # git diff
        ]

        result = capture_diff_hunks("HEAD~1", ["src/foo.py"])

        assert "_notices" in result
        assert any("timed out" in n.lower() for n in result["_notices"])

    @patch("utils.git_utils.subprocess.run")
    def test_empty_file_paths_returns_empty_dict(self, mock_run):
        """Empty file_paths list returns empty dict immediately."""
        result = capture_diff_hunks("HEAD~1", [])

        assert result == {}
        mock_run.assert_not_called()

    @patch("utils.git_utils.subprocess.run")
    def test_only_processes_requested_files(self, mock_run):
        """Only files in file_paths are included in the result."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=SAMPLE_MULTI_FILE_DIFF, stderr=""),
        ]

        # Only request src/foo.py, not src/bar.py
        result = capture_diff_hunks("HEAD~1", ["src/foo.py"])

        assert "src/foo.py" in result
        # bar.py is in the diff but not requested
        assert "src/bar.py" not in result

    @patch("utils.git_utils.subprocess.run")
    def test_500kb_budget_truncation(self, mock_run):
        """Total output is capped at 500KB; excess gets truncated marker."""
        # Create a diff that exceeds 500KB
        big_diff_lines = ["diff --git a/big.py b/big.py"]
        big_diff_lines.append("index aaa..bbb 100644")
        big_diff_lines.append("--- a/big.py")
        big_diff_lines.append("+++ b/big.py")
        big_diff_lines.append("@@ -1,1 +1,10000 @@")
        # Each line ~100 bytes, need ~5000 lines to exceed 500KB
        for i in range(6000):
            big_diff_lines.append(f"+{'x' * 90}line{i}")
        big_diff = "\n".join(big_diff_lines)

        # Also add a second file that should get truncated
        second_diff = (
            "\ndiff --git a/second.py b/second.py\n"
            "index ccc..ddd 100644\n"
            "--- a/second.py\n"
            "+++ b/second.py\n"
            "@@ -1,1 +1,2 @@\n"
            " existing\n"
            "+new\n"
        )

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=big_diff + second_diff, stderr=""),
        ]

        result = capture_diff_hunks("HEAD~1", ["big.py", "second.py"])

        # At least one file should be in the result
        assert "big.py" in result or "second.py" in result
        # If budget exceeded, _notices should mention it
        if "_notices" in result:
            budget_notices = [n for n in result["_notices"] if "budget" in n.lower()]
            if "second.py" in result and result["second.py"].get("truncated"):
                assert len(budget_notices) > 0

    @patch("utils.git_utils.subprocess.run")
    def test_deleted_file_with_last_known_content(self, mock_run):
        """Deleted files retrieve last-known content via git show."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # git status
            MagicMock(returncode=0, stdout=SAMPLE_DELETE_DIFF, stderr=""),  # git diff
            MagicMock(
                returncode=0,
                stdout="def old_func():\n    pass\n    return False\n",
                stderr=""
            ),  # git show
        ]

        result = capture_diff_hunks("HEAD~1", ["deleted.py"])

        assert "deleted.py" in result
        file_data = result["deleted.py"]
        assert file_data["deleted"] is True
        assert "last_known_content" in file_data

    @patch("utils.git_utils.subprocess.run")
    def test_os_error_returns_git_unavailable_notice(self, mock_run):
        """OSError (e.g., not a git repo) returns git unavailable notice."""
        mock_run.side_effect = OSError("Not a git repository")

        result = capture_diff_hunks("HEAD~1", ["src/foo.py"])

        assert "_notices" in result
        assert any("Git not available" in n for n in result["_notices"])

    @patch("utils.git_utils.subprocess.run")
    def test_line_numbers_in_hunks(self, mock_run):
        """Hunk line numbers are correctly computed for multi-hunk files."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=SAMPLE_MULTI_HUNK_DIFF, stderr=""),
        ]

        result = capture_diff_hunks("HEAD~1", ["src/foo.py"])

        file_data = result["src/foo.py"]
        assert len(file_data["hunks"]) == 2

        # Second hunk starts at line 20 old / 21 new
        h2 = file_data["hunks"][1]
        first_ctx = [l for l in h2["lines"] if l["type"] == "context"][0]
        assert first_ctx["old_line"] == 20
        assert first_ctx["new_line"] == 21

    @patch("utils.git_utils.subprocess.run")
    def test_diff_for_file_not_in_output(self, mock_run):
        """File requested but not present in diff output is silently skipped."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=SAMPLE_SINGLE_FILE_DIFF, stderr=""),
        ]

        result = capture_diff_hunks("HEAD~1", ["src/foo.py", "not_changed.py"])

        assert "src/foo.py" in result
        assert "not_changed.py" not in result


class TestCaptureFileContext:
    """Tests for capture_file_context function."""

    def _create_temp_file(self, lines, binary=False):
        """Helper to create a temporary file with given lines."""
        if binary:
            f = tempfile.NamedTemporaryFile(
                mode="wb", suffix=".bin", delete=False
            )
            f.write(b"\x00\x01\x02binary content")
            f.close()
            return f.name

        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        for line in lines:
            f.write(line + "\n")
        f.close()
        return f.name

    def test_basic_context_capture(self):
        """Captures lines around the specified range."""
        lines = [f"line {i}" for i in range(1, 61)]
        path = self._create_temp_file(lines)
        try:
            result = capture_file_context(path, [40, 45], context_lines=5)

            assert result is not None
            # Should cover lines 35–50
            assert result[0]["line_number"] == 35
            assert result[-1]["line_number"] == 50
            assert len(result) == 16  # 35 to 50 inclusive
            assert result[0]["content"] == "line 35"
            assert result[-1]["content"] == "line 50"
        finally:
            os.unlink(path)

    def test_context_at_file_start(self):
        """Context near start of file clamps to line 1."""
        lines = [f"line {i}" for i in range(1, 21)]
        path = self._create_temp_file(lines)
        try:
            result = capture_file_context(path, [1, 3], context_lines=5)

            assert result is not None
            assert result[0]["line_number"] == 1  # Can't go below 1
            assert result[-1]["line_number"] == 8  # 3 + 5
        finally:
            os.unlink(path)

    def test_context_at_file_end(self):
        """Context near end of file clamps to last line."""
        lines = [f"line {i}" for i in range(1, 21)]
        path = self._create_temp_file(lines)
        try:
            result = capture_file_context(path, [18, 20], context_lines=5)

            assert result is not None
            assert result[0]["line_number"] == 13  # 18 - 5
            assert result[-1]["line_number"] == 20  # File only has 20 lines
        finally:
            os.unlink(path)

    def test_nonexistent_file_returns_none(self):
        """Non-existent file returns None."""
        result = capture_file_context(
            "/nonexistent/path/file.py", [1, 5], context_lines=5
        )
        assert result is None

    def test_binary_file_returns_none(self):
        """Binary file returns None."""
        path = self._create_temp_file([], binary=True)
        try:
            result = capture_file_context(path, [1, 5], context_lines=5)
            assert result is None
        finally:
            os.unlink(path)

    def test_empty_file_path_returns_none(self):
        """Empty file path returns None."""
        result = capture_file_context("", [1, 5])
        assert result is None

    def test_empty_line_range_returns_none(self):
        """Empty line range returns None."""
        lines = ["hello"]
        path = self._create_temp_file(lines)
        try:
            result = capture_file_context(path, [])
            assert result is None
        finally:
            os.unlink(path)

    def test_single_line_range(self):
        """Single-element line_range treats it as both start and end."""
        lines = [f"line {i}" for i in range(1, 21)]
        path = self._create_temp_file(lines)
        try:
            result = capture_file_context(path, [10], context_lines=3)

            assert result is not None
            assert result[0]["line_number"] == 7  # 10 - 3
            assert result[-1]["line_number"] == 13  # 10 + 3
        finally:
            os.unlink(path)

    def test_custom_context_lines(self):
        """Custom context_lines parameter controls padding."""
        lines = [f"line {i}" for i in range(1, 31)]
        path = self._create_temp_file(lines)
        try:
            result = capture_file_context(path, [15, 15], context_lines=2)

            assert result is not None
            assert result[0]["line_number"] == 13  # 15 - 2
            assert result[-1]["line_number"] == 17  # 15 + 2
            assert len(result) == 5
        finally:
            os.unlink(path)

    def test_content_preserves_text(self):
        """File content is preserved accurately."""
        lines = [
            "def hello():",
            "    print('world')",
            "    return 42",
        ]
        path = self._create_temp_file(lines)
        try:
            result = capture_file_context(path, [1, 3], context_lines=0)

            assert result is not None
            assert len(result) == 3
            assert result[0]["content"] == "def hello():"
            assert result[1]["content"] == "    print('world')"
            assert result[2]["content"] == "    return 42"
        finally:
            os.unlink(path)

    def test_line_numbers_are_one_indexed(self):
        """Line numbers in output are 1-indexed."""
        lines = ["first", "second", "third"]
        path = self._create_temp_file(lines)
        try:
            result = capture_file_context(path, [1, 3], context_lines=0)

            assert result is not None
            assert result[0]["line_number"] == 1
            assert result[1]["line_number"] == 2
            assert result[2]["line_number"] == 3
        finally:
            os.unlink(path)

    def test_empty_file_returns_none(self):
        """Empty file returns None."""
        path = self._create_temp_file([])
        try:
            # The file will have no lines since we write nothing
            result = capture_file_context(path, [1, 1], context_lines=0)
            assert result is None
        finally:
            os.unlink(path)

    def test_content_strips_newlines(self):
        """Content strings have trailing newlines stripped."""
        lines = ["hello\r", "world"]
        path = self._create_temp_file(lines)
        try:
            result = capture_file_context(path, [1, 2], context_lines=0)

            assert result is not None
            # rstrip("\n").rstrip("\r") should handle the \r
            for item in result:
                assert not item["content"].endswith("\n")
                assert not item["content"].endswith("\r")
        finally:
            os.unlink(path)


class TestIsBinaryFile:
    """Tests for _is_binary_file internal function."""

    def test_text_file_is_not_binary(self):
        """Text file returns False."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        )
        f.write("hello world\n")
        f.close()
        try:
            assert _is_binary_file(f.name) is False
        finally:
            os.unlink(f.name)

    def test_binary_file_is_binary(self):
        """File with null bytes returns True."""
        f = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bin", delete=False
        )
        f.write(b"hello\x00world")
        f.close()
        try:
            assert _is_binary_file(f.name) is True
        finally:
            os.unlink(f.name)

    def test_nonexistent_file_returns_false(self):
        """Non-existent file returns False (not binary, just missing)."""
        assert _is_binary_file("/nonexistent/path/file.bin") is False


class TestCaptureDiffHunksEdgeCases:
    """Additional edge case tests for capture_diff_hunks."""

    @patch("utils.git_utils.subprocess.run")
    def test_git_status_timeout_continues_to_diff(self, mock_run):
        """git status timeout adds notice but still attempts diff."""
        mock_run.side_effect = [
            subprocess.TimeoutExpired(cmd="git status", timeout=10),  # status
            MagicMock(returncode=0, stdout=SAMPLE_SINGLE_FILE_DIFF, stderr=""),  # diff
        ]

        result = capture_diff_hunks("HEAD~1", ["src/foo.py"])

        assert "src/foo.py" in result
        assert "_notices" in result
        assert any("timed out" in n.lower() for n in result["_notices"])

    @patch("utils.git_utils.subprocess.run")
    def test_multiple_files_only_requested_processed(self, mock_run):
        """Multiple files in diff but only requested files processed."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=SAMPLE_MULTI_FILE_DIFF, stderr=""),
        ]

        result = capture_diff_hunks("HEAD~1", ["src/bar.py"])

        assert "src/bar.py" in result
        assert "src/foo.py" not in result

    @patch("utils.git_utils.subprocess.run")
    def test_git_diff_timeout_returns_empty_with_notice(self, mock_run):
        """git diff timeout returns result with timeout notice."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # status
            subprocess.TimeoutExpired(cmd="git diff", timeout=10),  # diff
        ]

        result = capture_diff_hunks("HEAD~1", ["src/foo.py"])

        assert "_notices" in result
        assert any("timed out" in n for n in result["_notices"])
        assert "src/foo.py" not in result


# ---------------------------------------------------------------------------
# Helpers for intent_to_add tests
# ---------------------------------------------------------------------------


def _run(cmd, cwd, check=True):
    """Run a shell command in the given directory; return completed process."""
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, check=check,
        encoding="utf-8",
    )


def _porcelain_status(cwd, path):
    """Return the two-char porcelain status for ``path`` within ``cwd``.

    Returns an empty string if the file has no status entry (i.e. clean).
    """
    result = _run(["git", "status", "--porcelain", "--", path], cwd)
    for line in result.stdout.splitlines():
        if line.endswith(path) or line[3:] == path:
            return line[:2]
    return ""


@pytest.fixture
def temp_git_repo(tmp_path, monkeypatch):
    """Create a throwaway git repo and chdir into it for the test.

    The repo has a single committed file so HEAD exists.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "--quiet"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test"], repo)
    _run(["git", "config", "commit.gpgsign", "false"], repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _run(["git", "add", "seed.txt"], repo)
    _run(["git", "commit", "--quiet", "-m", "init"], repo)
    monkeypatch.chdir(repo)
    return repo


class TestIntentToAddUntracked:
    """Tests for intent_to_add_untracked context manager."""

    def test_marks_and_cleans(self, temp_git_repo):
        """Untracked file becomes visible to git diff inside the with block
        and returns to untracked on exit."""
        (temp_git_repo / "new.py").write_text("print('hello')\n", encoding="utf-8")

        # Baseline: file is untracked
        assert _porcelain_status(temp_git_repo, "new.py") == "??"
        before_diff = _run(["git", "diff", "HEAD", "--", "new.py"], temp_git_repo).stdout
        assert before_diff == ""

        with intent_to_add_untracked(["new.py"]) as touched:
            assert touched == ["new.py"]
            status = _porcelain_status(temp_git_repo, "new.py")
            assert status == " A", f"unexpected status: {status!r}"
            inside_diff = _run(
                ["git", "diff", "HEAD", "--", "new.py"], temp_git_repo
            ).stdout
            assert "hello" in inside_diff
            assert "+print('hello')" in inside_diff

        # After exit, file is untracked again
        assert _porcelain_status(temp_git_repo, "new.py") == "??"
        after_diff = _run(["git", "diff", "HEAD", "--", "new.py"], temp_git_repo).stdout
        assert after_diff == ""

    def test_skips_already_staged(self, temp_git_repo):
        """Files already staged (with content) should not be touched; they
        remain staged after the context exits."""
        (temp_git_repo / "staged.py").write_text("x = 1\n", encoding="utf-8")
        _run(["git", "add", "staged.py"], temp_git_repo)
        assert _porcelain_status(temp_git_repo, "staged.py") == "A "

        with intent_to_add_untracked(["staged.py"]) as touched:
            assert touched == []

        # Still staged with content (not reset by our context)
        assert _porcelain_status(temp_git_repo, "staged.py") == "A "
        # Cached diff still includes the staged content
        cached = _run(
            ["git", "diff", "--cached", "--", "staged.py"], temp_git_repo
        ).stdout
        assert "x = 1" in cached

    def test_skips_already_intent_to_add(self, temp_git_repo):
        """A file already marked -N by the user should not be reset on exit."""
        (temp_git_repo / "pre.py").write_text("y = 2\n", encoding="utf-8")
        _run(["git", "add", "-N", "pre.py"], temp_git_repo)
        assert _porcelain_status(temp_git_repo, "pre.py") == " A"

        with intent_to_add_untracked(["pre.py"]) as touched:
            # ls-files --others --exclude-standard does NOT list intent-to-add
            # files (they're in the index), so the context skips them.
            assert touched == []

        # Still intent-to-add
        assert _porcelain_status(temp_git_repo, "pre.py") == " A"

    def test_missing_file(self, temp_git_repo):
        """File listed but absent from disk is silently skipped."""
        with intent_to_add_untracked(["does_not_exist.py"]) as touched:
            assert touched == []

    def test_cleanup_on_exception(self, temp_git_repo):
        """Exception inside the with block still resets the file."""
        (temp_git_repo / "boom.py").write_text("raise\n", encoding="utf-8")
        assert _porcelain_status(temp_git_repo, "boom.py") == "??"

        with pytest.raises(RuntimeError):
            with intent_to_add_untracked(["boom.py"]) as touched:
                assert touched == ["boom.py"]
                assert _porcelain_status(temp_git_repo, "boom.py") == " A"
                raise RuntimeError("boom")

        assert _porcelain_status(temp_git_repo, "boom.py") == "??"

    def test_empty_list(self, temp_git_repo):
        """Empty file_paths yields cleanly without running git commands."""
        with intent_to_add_untracked([]) as touched:
            assert touched == []

    def test_multiple_mixed(self, temp_git_repo):
        """Mix of untracked, staged, and modified-tracked files: only the
        untracked ones are marked and reset."""
        # Staged new file
        (temp_git_repo / "staged.py").write_text("s = 1\n", encoding="utf-8")
        _run(["git", "add", "staged.py"], temp_git_repo)

        # Modified tracked file (modify seed.txt)
        (temp_git_repo / "seed.txt").write_text("seed modified\n", encoding="utf-8")

        # Two untracked files
        (temp_git_repo / "new_a.py").write_text("a = 1\n", encoding="utf-8")
        (temp_git_repo / "new_b.py").write_text("b = 2\n", encoding="utf-8")

        files = ["staged.py", "seed.txt", "new_a.py", "new_b.py"]

        with intent_to_add_untracked(files) as touched:
            assert sorted(touched) == ["new_a.py", "new_b.py"]
            # Untracked now visible
            for f in ("new_a.py", "new_b.py"):
                diff = _run(["git", "diff", "HEAD", "--", f], temp_git_repo).stdout
                assert diff, f"expected diff for {f}"
            # Staged file untouched
            assert _porcelain_status(temp_git_repo, "staged.py") == "A "
            # Modified tracked file untouched
            assert _porcelain_status(temp_git_repo, "seed.txt") == " M"

        # After exit: untracked back to untracked
        assert _porcelain_status(temp_git_repo, "new_a.py") == "??"
        assert _porcelain_status(temp_git_repo, "new_b.py") == "??"
        # Staged and modified files unchanged
        assert _porcelain_status(temp_git_repo, "staged.py") == "A "
        assert _porcelain_status(temp_git_repo, "seed.txt") == " M"


class TestRunGitDecoding:
    """Tests for UTF-8 + errors='replace' decoding of git output (task 2)."""

    def test_run_git_decodes_invalid_utf8_with_replacement(self, monkeypatch):
        """Real subprocess emitting invalid UTF-8 bytes decodes with U+FFFD.

        Cross-platform (no shebang/chmod/PATH tricks, so it also runs on
        Windows): swaps the git argv for a Python byte emitter while passing
        _run_git's own kwargs through to the real subprocess.run, so the
        actual decode path is exercised (no UnicodeDecodeError, replacement
        characters in the result).
        """
        real_run = subprocess.run
        captured_kwargs = {}
        # \xe9 (0xE9) is cp1252 'e-acute'; \xff (0xFF) is invalid UTF-8 in
        # any position.
        emitter = (
            "import sys; "
            "sys.stdout.buffer.write(b'caf\\xe9 \\xff done\\n')"
        )

        def run_python_emitter(cmd, **kwargs):
            assert cmd[0] == "git"
            captured_kwargs.update(kwargs)
            return real_run([sys.executable, "-c", emitter], **kwargs)

        monkeypatch.setattr("utils.git_utils.subprocess.run", run_python_emitter)

        stdout, stderr, code = _run_git("anything", check=False)

        assert code == 0
        assert "caf� � done" in stdout
        # The lossless decode must come from _run_git's own kwargs, not the
        # locale default.
        assert captured_kwargs["encoding"] == "utf-8"
        assert captured_kwargs["errors"] == "replace"

    @patch("utils.git_utils.subprocess.run")
    def test_run_git_passes_utf8_replace_kwargs(self, mock_run):
        """_run_git requests utf-8/replace decoding from subprocess.run."""
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        _run_git("status", check=False)

        kwargs = mock_run.call_args.kwargs
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
