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
    _per_file_line_cap,
    _run_git,
    _split_diff_by_file,
    baseline_covers,
    capture_diff_hunks,
    capture_file_context,
    classify_review_paths,
    compile_path_glob,
    count_review_groups,
    get_branch_name,
    get_current_head,
    get_diff_since_ref,
    get_file_diff,
    get_files_changed_since_ref,
    get_modified_files,
    get_staged_diff,
    get_staged_files,
    ignored_paths,
    intent_to_add_untracked,
    is_clean_working_tree,
    list_untracked_files,
    list_worktree_untracked_files,
    order_review_paths,
    stage_files,
    subtract_baseline,
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

        'M ' is modified-in-index; the worktree-only form ' M', whose leading
        space used to be eaten, has its own regression test below.
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
        """Multiple files with different statuses all appear."""
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
        """Empty lines in output are filtered out."""
        mock_run_git.return_value = ("M  file1.py\n\nM  file2.py\n", "", 0)

        files = get_modified_files()

        assert len(files) == 2
        assert "file1.py" in files
        assert "file2.py" in files

    @patch("utils.git_utils._run_git")
    def test_worktree_only_change_on_first_line_keeps_its_first_character(
        self, mock_run_git
    ):
        """A leading ' M' on the FIRST line must not shift the filename.

        Regression: the implementation used to call ``stdout.strip()`` on the
        whole output, which removes the leading space of the first line only.
        ``line[3:]`` then started one character late and returned
        ``'rc/auth.py'`` for ``' M src/auth.py'`` -- a path that does not exist,
        which then reached state.json and the review prompts. Every later line
        was unaffected, which is why the bug survived.
        """
        mock_run_git.return_value = (
            (
                " M src/auth.py\n"
                "?? .multi-llm/\n"
                "?? src/security.py\n"
            ),
            "",
            0,
        )

        files = get_modified_files()

        assert files == ["src/auth.py", ".multi-llm/", "src/security.py"]

    @patch("utils.git_utils._run_git")
    def test_every_worktree_only_status_column_is_preserved(self, mock_run_git):
        """The worktree column (' M', ' D', ' A') survives on any line."""
        mock_run_git.return_value = (
            (
                " M first.py\n"
                " D second.py\n"
                " M third.py\n"
            ),
            "",
            0,
        )

        files = get_modified_files()

        assert files == ["first.py", "second.py", "third.py"]

    @patch("utils.git_utils._run_git")
    def test_renamed_file_returns_destination_path(self, mock_run_git):
        """A rename is `XY old -> new`; only the destination exists on disk."""
        mock_run_git.return_value = ("R  src/old.py -> src/new.py\n", "", 0)

        files = get_modified_files()

        assert files == ["src/new.py"]

    @patch("utils.git_utils._run_git")
    def test_renamed_and_modified_file_returns_destination_path(self, mock_run_git):
        """'RM' (renamed in index, modified in worktree) is a rename too."""
        mock_run_git.return_value = (
            "RM src/old.py -> src/new.py\n M other.py\n", "", 0
        )

        files = get_modified_files()

        assert files == ["src/new.py", "other.py"]

    @patch("utils.git_utils._run_git")
    def test_copied_file_returns_destination_path(self, mock_run_git):
        """'C' (copy detection) uses the same `old -> new` rendering."""
        mock_run_git.return_value = ("C  src/orig.py -> src/copy.py\n", "", 0)

        files = get_modified_files()

        assert files == ["src/copy.py"]

    @patch("utils.git_utils._run_git")
    def test_renamed_file_with_quoted_paths_returns_destination(self, mock_run_git):
        """Git quotes a path containing a space, on either side of the arrow.

        The quoting is what keeps the separator unambiguous, so the destination
        is still recoverable when the source itself contains ' -> '.
        """
        mock_run_git.return_value = (
            'R  "src/old name.py" -> "src/new name.py"\n', "", 0
        )
        assert get_modified_files() == ["src/new name.py"]

        mock_run_git.return_value = ('R  "src/a -> b.py" -> src/c.py\n', "", 0)
        assert get_modified_files() == ["src/c.py"]

        mock_run_git.return_value = ('R  src/a.py -> "src/b -> c.py"\n', "", 0)
        assert get_modified_files() == ["src/b -> c.py"]

    @patch("utils.git_utils._run_git")
    def test_truncated_lines_are_skipped(self, mock_run_git):
        """A line too short to carry a path yields no entry (never an empty one)."""
        mock_run_git.return_value = (" M \nM\n M kept.py\n", "", 0)

        files = get_modified_files()

        assert files == ["kept.py"]

    def test_an_untracked_directory_stays_one_collapsed_entry(self, temp_git_repo):
        """An untracked *directory* is censused as one ``plans/`` entry.

        Expanding it (``-uall``) makes the census disagree with every baseline
        already stored in a state file and, in a repo with a large untracked
        tree, writes all of it into ``tracked_files``. The compact entry is what
        ``subtract_baseline`` knows how to read.
        """
        untracked_dir = temp_git_repo / "plans"
        untracked_dir.mkdir()
        (untracked_dir / "one.md").write_text("one\n", encoding="utf-8")
        (untracked_dir / "two.md").write_text("two\n", encoding="utf-8")

        files = get_modified_files()

        assert files == ["plans/"]

    @patch("utils.git_utils._run_git")
    def test_a_path_with_a_space_is_censused_without_its_quotes(self, mock_run_git):
        """F6: the census is what gets STORED, so a quoted entry would reach
        `tracked_files` with its quotes and `review-code` would diff a path
        that does not exist."""
        mock_run_git.return_value = ('?? "sp ace.txt"\n M plain.txt\n', "", 0)

        assert get_modified_files() == ["sp ace.txt", "plain.txt"]

    @patch("utils.git_utils._run_git")
    def test_a_c_escaped_path_is_decoded_back_to_its_real_name(self, mock_run_git):
        """Git escapes a non-printable byte as three-digit octal, one per UTF-8
        byte, and the control characters by name."""
        mock_run_git.return_value = ('?? "caf\\303\\251.md"\n?? "a\\tb.txt"\n', "", 0)

        assert get_modified_files() == ["café.md", "a\tb.txt"]

    @patch("utils.git_utils._run_git")
    def test_a_bare_backslash_path_is_left_alone(self, mock_run_git):
        """Only a *quoted* field is escaped; an unquoted backslash is literal."""
        mock_run_git.return_value = (" M src\\weird.py\n", "", 0)

        assert get_modified_files() == ["src\\weird.py"]


class TestListUntrackedFiles:
    """The expansion `finalize_tracking` applies to a directory the phase
    created — the one entry the compact census cannot hand to a consumer."""

    def test_it_names_the_files_under_an_untracked_directory(self, temp_git_repo):
        (temp_git_repo / "src").mkdir()
        (temp_git_repo / "src" / "a.js").write_text("a\n", encoding="utf-8")
        (temp_git_repo / "src" / "b.js").write_text("b\n", encoding="utf-8")

        assert sorted(list_untracked_files("src/")) == ["src/a.js", "src/b.js"]

    def test_a_path_with_a_space_comes_back_raw(self, temp_git_repo):
        """``-z`` output carries no quoting to undo."""
        (temp_git_repo / "we ird").mkdir()
        (temp_git_repo / "we ird" / "g h.md").write_text("g\n", encoding="utf-8")

        assert list_untracked_files("we ird/") == ["we ird/g h.md"]

    def test_a_directory_that_is_not_there_yields_nothing(self, temp_git_repo):
        assert list_untracked_files("gone/") == []


class TestSubtractBaseline:
    """A baseline entry ending in ``/`` covers every path beneath it."""

    def test_a_directory_baseline_entry_covers_the_files_under_it(self):
        current = [
            "plans/game.md",
            "plans/game/tasks/tasks.md",
            "saved/session.json",
            "src/main.py",
        ]
        baseline = [".claude/", ".multi-llm/", "plans/", "saved/"]

        assert subtract_baseline(current, baseline) == ["src/main.py"]

    def test_a_file_baseline_entry_still_matches_exactly(self):
        """A plain entry must not become a prefix: `src/a.py` is not `src/ab.py`."""
        current = ["src/a.py", "src/ab.py", "src/a.pyc"]

        assert subtract_baseline(current, ["src/a.py"]) == ["src/ab.py", "src/a.pyc"]

    def test_the_directory_itself_is_covered_with_and_without_the_slash(self):
        assert subtract_baseline(["plans/", "plans"], ["plans/"]) == []

    def test_input_order_is_preserved(self):
        assert subtract_baseline(["z.py", "a.py"], []) == ["z.py", "a.py"]

    def test_a_quoted_directory_entry_covers_its_quoted_children(self):
        """Git quotes a path containing a space on both sides of the comparison."""
        current = ['"my plans/one.md"', "src/main.py"]

        assert subtract_baseline(current, ['"my plans/"']) == ["src/main.py"]

    def test_baseline_covers_answers_for_a_single_path(self):
        assert baseline_covers("plans/game.md", ["plans/"])
        assert not baseline_covers("src/main.py", ["plans/"])


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

    @patch("utils.git_utils._run_git")
    def test_leading_whitespace_in_a_filename_is_preserved(self, mock_run_git):
        """`--name-only` never quotes a leading space, so stripping loses it.

        Same class as the get_modified_files regression: a whole-output
        ``.strip()`` only mangles the first (and last) line, so the damage is
        invisible in every multi-file test.
        """
        mock_run_git.return_value = (" leading.py\nnormal.py\n", "", 0)

        assert get_staged_files() == [" leading.py", "normal.py"]


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

    @patch("utils.git_utils._run_git")
    def test_leading_whitespace_in_a_filename_is_preserved(self, mock_run_git):
        """`--name-only` paths are unquoted, so only the newline may be dropped."""
        mock_run_git.return_value = (" leading.py\nnormal.py\n", "", 0)

        assert get_files_changed_since_ref("HEAD~1") == [" leading.py", "normal.py"]


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

    def test_truncation_reports_the_real_total_line_count(self):
        """truncated_at is accompanied by total_lines, so a reader can show
        'first N of total_lines' instead of 'first N of ?'."""
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

        result = _parse_unified_diff(diff_text, max_lines=200)

        assert result.get("truncated_at") == 200
        assert result.get("total_lines") == 250

    def test_respects_a_custom_max_lines(self):
        """max_lines is the cap _parse_unified_diff actually enforces, not
        the module's flat default — capture_diff_hunks scales it per call."""
        diff_lines = [
            "diff --git a/big.py b/big.py",
            "index aaa..bbb 100644",
            "--- a/big.py",
            "+++ b/big.py",
            "@@ -1,50 +1,50 @@",
        ]
        for i in range(1, 51):
            diff_lines.append(f"+line {i}")
        diff_text = "\n".join(diff_lines)

        result = _parse_unified_diff(diff_text, max_lines=1000)

        assert "truncated_at" not in result
        assert "total_lines" not in result
        total_captured = sum(len(h["lines"]) for h in result["hunks"])
        assert total_captured == 50

    def test_untruncated_result_has_no_total_lines_key(self):
        """total_lines is only meaningful (and only present) alongside
        truncated_at — an untruncated diff has nothing to reconcile."""
        result = _parse_unified_diff(SAMPLE_SINGLE_FILE_DIFF)
        assert "truncated_at" not in result
        assert "total_lines" not in result

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


class TestPerFileLineCap:
    """Tests for _per_file_line_cap — scales the per-file diff cap by how
    many files share the total byte budget, instead of a flat 200 lines."""

    def test_single_file_gets_far_more_than_the_old_flat_cap(self):
        """A lone file (the common whole-file-review case) is allowed most
        of _MAX_TOTAL_BYTES, not the 200-line floor."""
        cap = _per_file_line_cap(1)
        assert cap > 200
        # 1393 lines (the racing-game index.html diff that motivated this)
        # must fit comfortably under a single-file cap.
        assert cap > 1393

    def test_many_files_fall_back_to_the_floor(self):
        """A large fan-out still gets at least the historical 200-line
        floor per file, however many files are being diffed."""
        assert _per_file_line_cap(500) == 200

    def test_cap_shrinks_as_file_count_grows(self):
        """More files sharing the same byte budget means a smaller
        per-file share."""
        assert _per_file_line_cap(2) > _per_file_line_cap(10)

    def test_zero_files_falls_back_to_the_flat_default(self):
        """Degenerate input (no files) must not divide by zero."""
        assert _per_file_line_cap(0) == 200


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
    def test_single_file_review_is_not_capped_at_200_lines(self, mock_run):
        """A whole-file review of one file (well under the byte budget)
        gets its full diff, not the old flat 200-line cap — this is the
        1,393-insertion index.html case the per-file cap used to gut."""
        diff_lines = [
            "diff --git a/index.html b/index.html",
            "index aaa..bbb 100644",
            "--- a/index.html",
            "+++ b/index.html",
            "@@ -1,1 +1,1393 @@",
        ]
        for i in range(1, 1394):
            diff_lines.append(f"+line {i}")
        diff_text = "\n".join(diff_lines)

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout=diff_text, stderr=""),
        ]

        result = capture_diff_hunks("HEAD~1", ["index.html"])

        file_data = result["index.html"]
        assert "truncated_at" not in file_data
        captured = sum(len(h["lines"]) for h in file_data["hunks"])
        assert captured == 1393

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


class TestIgnoredPaths:
    """`ignored_paths` (2026-10-02, #22) asks `git check-ignore` which of the
    paths it is handed git ignores. W7 (adversarial review, 2026-10-02): one
    path outside the repository made the whole call exit 128, and the function
    answered "nothing is ignored" for every path beside it — the one answer
    that re-admits a transcript to `tracked_files`."""

    def test_an_outside_path_does_not_disable_the_answer_for_the_rest(
        self, temp_git_repo
    ):
        (temp_git_repo / ".gitignore").write_text("logs/\n", encoding="utf-8")
        (temp_git_repo / "logs").mkdir()
        (temp_git_repo / "logs" / "run.jsonl").write_text("{}\n", encoding="utf-8")
        outside = str(temp_git_repo.parent / "elsewhere.txt")

        answer = ignored_paths(["logs/run.jsonl", outside, "seed.txt"],
                               cwd=str(temp_git_repo))

        assert answer == {"logs/run.jsonl"}

    def test_an_outside_path_alone_is_simply_not_ignored(self, temp_git_repo):
        outside = str(temp_git_repo.parent / "elsewhere.txt")
        assert ignored_paths([outside], cwd=str(temp_git_repo)) == set()

    def test_outside_a_repository_nothing_is_ignored(self, tmp_path):
        bare = tmp_path / "not-a-repo"
        bare.mkdir()
        assert ignored_paths(["x"], cwd=str(bare)) == set()


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

    def test_never_writes_the_real_index(self, temp_git_repo, monkeypatch):
        """#23: the mark lives in a private index copy exported as
        GIT_INDEX_FILE. The user's `.git/index` is byte-identical throughout, the
        variable is restored, and no copy is left in the git dir."""
        import os

        monkeypatch.delenv("GIT_INDEX_FILE", raising=False)
        (temp_git_repo / "new.py").write_text("print('hello')\n", encoding="utf-8")
        real = temp_git_repo / ".git" / "index"
        before = real.read_bytes()

        with intent_to_add_untracked(["new.py"]) as touched:
            assert touched == ["new.py"]
            assert os.environ["GIT_INDEX_FILE"] != str(real)
            assert real.read_bytes() == before
            assert "hello" in _run(["git", "diff", "HEAD", "--", "new.py"], temp_git_repo).stdout
            # What a user's own terminal sees (no inherited variable): untracked.
            clean_env = {k: v for k, v in os.environ.items() if k != "GIT_INDEX_FILE"}
            status = subprocess.run(
                ["git", "status", "--porcelain", "--", "new.py"],
                cwd=str(temp_git_repo), capture_output=True, text=True, env=clean_env,
            ).stdout
            assert status.startswith("??")

        assert "GIT_INDEX_FILE" not in os.environ
        assert real.read_bytes() == before
        assert not [p for p in (temp_git_repo / ".git").iterdir()
                    if p.name.startswith("multi-llm-index.")]

    def test_a_relative_preset_index_var_resolves_against_cwd(
        self, temp_git_repo, monkeypatch, tmp_path, capsys
    ):
        """A git hook exports `GIT_INDEX_FILE=.git/index`; the copy's source
        is that path relative to where git runs (`cwd`), not to this process's
        own directory (adversarial review, 2026-10-02: resolved against the
        process cwd, the target dir did not exist, `git add -N` failed on its
        lock file and the file stayed invisible with only a warning)."""
        import os

        (temp_git_repo / "new.py").write_text("print('hello')\n", encoding="utf-8")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setenv("GIT_INDEX_FILE", ".git/index")

        with intent_to_add_untracked(["new.py"], cwd=str(temp_git_repo)) as touched:
            assert touched == ["new.py"]
            copy = os.environ["GIT_INDEX_FILE"]
            assert os.path.isabs(copy)
            assert Path(copy).parent == temp_git_repo / ".git"
            assert "hello" in _run(["git", "diff", "HEAD", "--", "new.py"], temp_git_repo).stdout

        assert os.environ["GIT_INDEX_FILE"] == ".git/index"
        assert "could not mark untracked files" not in capsys.readouterr().err
        assert not [p for p in (temp_git_repo / ".git").iterdir()
                    if p.name.startswith("multi-llm-index.")]

    def test_sweeps_the_copies_a_killed_run_left_behind(self, temp_git_repo, monkeypatch):
        """SIGKILL skips the `finally`, so a copy named after a dead pid (and
        its `.lock`) stayed in `.git` forever (adversarial review, 2026-10-02).
        Entry sweeps those; copies of this process and of another live
        process are left alone."""
        import os

        monkeypatch.delenv("GIT_INDEX_FILE", raising=False)
        git_dir = temp_git_repo / ".git"
        from utils.process_control import process_alive

        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        dead_pid = child.pid
        assert not process_alive(dead_pid)
        gone = [
            git_dir / f"multi-llm-index.{dead_pid}.111",
            git_dir / f"multi-llm-index.{dead_pid}.111.lock",
            git_dir / f"multi-llm-index.{dead_pid}.222",
        ]
        kept = [
            git_dir / f"multi-llm-index.{os.getpid()}.333",
            git_dir / f"multi-llm-index.1.444",  # init: alive, not ours
            git_dir / "multi-llm-index.notapid",
        ]
        for path in gone + kept:
            path.write_bytes(b"x")
        (temp_git_repo / "new.py").write_text("x = 1\n", encoding="utf-8")

        with intent_to_add_untracked(["new.py"]) as touched:
            assert touched == ["new.py"]

        for path in gone:
            assert not path.exists(), path.name
        for path in kept:
            assert path.exists(), path.name
            path.unlink()

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


class TestGetModifiedFilesAgainstRealGit:
    """The porcelain parse, driven by a real `git status` rather than a mock.

    The mocked tests pin the parse; these pin the *format assumption* behind it,
    so a change in how git renders a worktree-only change or a rename is caught
    here instead of being encoded into both sides of a mock.
    """

    def test_worktree_modification_is_reported_with_its_full_path(
        self, temp_git_repo
    ):
        """The reproduction from the bug report, end to end.

        ``git status --porcelain`` sorts by path, so ``src/auth.py`` (a
        worktree-only ``' M'``) lands on the first line, which is the only line
        a whole-output ``.strip()`` damages.
        """
        src = temp_git_repo / "src"
        src.mkdir()
        (src / "auth.py").write_text("original\n", encoding="utf-8")
        _run(["git", "add", "src/auth.py"], temp_git_repo)
        _run(["git", "commit", "--quiet", "-m", "add auth"], temp_git_repo)

        (src / "auth.py").write_text("modified\n", encoding="utf-8")
        (temp_git_repo / ".multi-llm").mkdir()
        (temp_git_repo / ".multi-llm" / "state.json").write_text("{}", encoding="utf-8")
        (src / "security.py").write_text("new\n", encoding="utf-8")

        porcelain = _run(["git", "status", "--porcelain"], temp_git_repo).stdout
        assert porcelain.startswith(" M src/auth.py"), porcelain

        files = get_modified_files()

        assert files == ["src/auth.py", ".multi-llm/", "src/security.py"], files
        assert all(
            (temp_git_repo / path.rstrip("/")).exists() for path in files
        ), files

    def test_a_rename_reports_the_path_that_exists(self, temp_git_repo):
        """`R  old -> new` must yield the destination, the only path on disk."""
        (temp_git_repo / "old.py").write_text("content\n", encoding="utf-8")
        _run(["git", "add", "old.py"], temp_git_repo)
        _run(["git", "commit", "--quiet", "-m", "add old"], temp_git_repo)
        _run(["git", "mv", "old.py", "new.py"], temp_git_repo)

        porcelain = _run(["git", "status", "--porcelain"], temp_git_repo).stdout
        assert porcelain.strip() == "R  old.py -> new.py", porcelain

        assert get_modified_files() == ["new.py"]


# ---------------------------------------------------------------------------
# The review-scope classifier (review-scope plan §2 D2.1): one set of rules for
# the prompt's file list and the scope manifest, run in a real repository so
# the `linguist-generated` attribute comes from a real `.gitattributes`.
# ---------------------------------------------------------------------------


@pytest.fixture
def attributed_repo(temp_git_repo):
    """``temp_git_repo`` with a ``.gitattributes`` marking a generated tree.

    One pattern reaches INTO ``tests/fixtures/`` on purpose, so a path there can
    carry the attribute and prove generated outranks fixtures.
    """
    (temp_git_repo / ".gitattributes").write_text(
        "mirror/** linguist-generated=true -diff\n"
        "tests/fixtures/gen/** linguist-generated\n"
        "notgen/** -linguist-generated\n",
        encoding="utf-8",
    )
    return temp_git_repo


class TestClassifyReviewPaths:
    def test_a_linguist_generated_path_under_fixtures_is_generated(self, attributed_repo):
        groups = classify_review_paths([
            "tests/fixtures/gen/big.json",   # attribute AND fixtures segment
            "mirror/utils/x.py",             # attribute on a would-be source file
            "mirror/tests/test_x.py",        # attribute on a would-be test
        ])
        assert groups == {
            "tests/fixtures/gen/big.json": "generated",
            "mirror/utils/x.py": "generated",
            "mirror/tests/test_x.py": "generated",
        }

    def test_an_unset_attribute_is_not_generated(self, attributed_repo):
        assert classify_review_paths(["notgen/a.py"]) == {"notgen/a.py": "source"}

    def test_a_golden_path_is_a_fixture_not_a_test(self, attributed_repo):
        groups = classify_review_paths([
            "tests/golden/prompts/x.txt",
            "skills/multi-llm/tests/fixtures/pi/a.stdout.txt",
        ])
        assert set(groups.values()) == {"fixtures"}

    def test_tests_and_source(self, attributed_repo):
        assert classify_review_paths(["tests/test_x.py", "utils/x.py"]) == {
            "tests/test_x.py": "tests",
            "utils/x.py": "source",
        }

    def test_segments_match_whole_directory_names_at_any_depth(self, attributed_repo):
        groups = classify_review_paths([
            "skills/multi-llm/tests/x.py",   # nested tests/ segment
            "pkg/test/helpers.py",           # test/ segment
            "latest/x.py",                   # "test" inside a name, not a segment
            "contests/x.py",
            "contests/fixtures/y.json",      # not a tests/fixtures segment
        ])
        assert groups == {
            "skills/multi-llm/tests/x.py": "tests",
            "pkg/test/helpers.py": "tests",
            "latest/x.py": "source",
            "contests/x.py": "source",
            "contests/fixtures/y.json": "source",
        }

    def test_test_file_names_are_read_from_the_name_only(self, attributed_repo):
        groups = classify_review_paths([
            "src/test_util.py", "go/x_test.go", "web/a.test.ts", "web/b.spec.js",
            "test_data/readme.md",            # a directory name, not a file name
            "src/contest.py",
        ])
        assert groups == {
            "src/test_util.py": "tests",
            "go/x_test.go": "tests",
            "web/a.test.ts": "tests",
            "web/b.spec.js": "tests",
            "test_data/readme.md": "source",
            "src/contest.py": "source",
        }

    def test_outside_a_repository_nothing_is_generated_and_the_rest_still_applies(
        self, tmp_path, monkeypatch, capsys
    ):
        """`git check-attr` fails there: one warning, no generated group, and the
        glob and segment rules classify every path as usual."""
        outside = tmp_path / "not-a-repo"
        outside.mkdir()
        monkeypatch.chdir(outside)
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))

        groups = classify_review_paths([
            "mirror/utils/x.py", "tests/fixtures/gen/big.json",
            "tests/test_x.py", "utils/x.py",
        ])

        assert groups == {
            "mirror/utils/x.py": "source",
            "tests/fixtures/gen/big.json": "fixtures",
            "tests/test_x.py": "tests",
            "utils/x.py": "source",
        }
        err = capsys.readouterr().err
        assert err.count("git check-attr linguist-generated failed") == 1

    def test_configured_globs_run_after_the_attribute_and_before_tests(
        self, attributed_repo
    ):
        collapse = {"generated": ["**/*.lock", "dist/**"], "fixtures": ["**/data/**"]}
        groups = classify_review_paths(
            ["poetry.lock", "dist/app.js", "tests/data/case.json", "tests/fixtures/x.json"],
            collapse,
        )
        assert groups == {
            "poetry.lock": "generated",
            "dist/app.js": "generated",
            "tests/data/case.json": "fixtures",
            # The fixture rule was REPLACED by this config, so a
            # tests/fixtures/ path now falls through to the tests rule.
            "tests/fixtures/x.json": "tests",
        }

    def test_a_key_left_out_keeps_its_default(self, attributed_repo):
        groups = classify_review_paths(
            ["poetry.lock", "tests/fixtures/x.json"], {"generated": ["**/*.lock"]}
        )
        assert groups == {"poetry.lock": "generated", "tests/fixtures/x.json": "fixtures"}

    def test_duplicates_and_empty_paths_are_dropped(self, attributed_repo):
        assert classify_review_paths(["a.py", "", "a.py"]) == {"a.py": "source"}

    def test_count_review_groups_names_every_group(self):
        assert count_review_groups({"a.py": "source", "t/test_a.py": "tests"}) == {
            "source": 1, "tests": 1, "fixtures": 0, "generated": 0,
        }


class TestCompilePathGlob:
    @pytest.mark.parametrize(
        "pattern, path, expected",
        [
            ("**/tests/fixtures/**", "tests/fixtures/a.json", True),
            ("**/tests/fixtures/**", "a/b/tests/fixtures/c/d.json", True),
            ("**/tests/fixtures/**", "contests/fixtures/a.json", False),
            ("**/*.lock", "poetry.lock", True),
            ("*.lock", "web/yarn.lock", True),
            ("dist/**", "dist/a/b.js", True),
            ("dist/**", "src/dist/a.js", False),
            ("src/*.py", "src/a/b.py", False),
            ("a/**/b.txt", "a/b.txt", True),
            ("a/**/b.txt", "a/x/y/b.txt", True),
            ("file?.py", "file1.py", True),
            ("[!a]x.py", "bx.py", True),
            ("[!a]x.py", "ax.py", False),
        ],
    )
    def test_gitignore_like_semantics(self, pattern, path, expected):
        assert bool(compile_path_glob(pattern).fullmatch(path)) is expected


class TestOrderReviewPaths:
    PATHS = [
        "skills/multi-llm/tests/fixtures/opencode/a.json",
        "skills/multi-llm/tests/test_git_utils.py",
        "mirror/utils/git_utils.py",
        "skills/multi-llm/utils/z.py",
        "skills/multi-llm/utils/git_utils.py",
        "skills/multi-llm/tests/golden/b.txt",
    ]
    EXPECTED = [
        "skills/multi-llm/utils/git_utils.py",
        "skills/multi-llm/utils/z.py",
        "skills/multi-llm/tests/test_git_utils.py",
        "skills/multi-llm/tests/fixtures/opencode/a.json",
        "skills/multi-llm/tests/golden/b.txt",
        "mirror/utils/git_utils.py",
    ]

    def test_source_tests_fixtures_generated_then_by_path(self, attributed_repo):
        assert order_review_paths(self.PATHS) == self.EXPECTED

    def test_every_permutation_gives_the_same_list(self, attributed_repo):
        """The tracked-files branch de-duplicates through a set, so its order
        differs run to run; the rendered list must not."""
        import itertools

        results = {
            tuple(order_review_paths(permutation))
            for permutation in itertools.permutations(self.PATHS)
        }
        assert results == {tuple(self.EXPECTED)}


class TestListWorktreeUntrackedFiles:
    def test_it_is_worktree_wide_and_root_relative_from_a_subdirectory(
        self, temp_git_repo, monkeypatch
    ):
        (temp_git_repo / "a").mkdir()
        (temp_git_repo / "a" / "new.py").write_text("x\n", encoding="utf-8")
        (temp_git_repo / "b").mkdir()
        (temp_git_repo / "b" / "other.txt").write_text("y\n", encoding="utf-8")
        (temp_git_repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
        (temp_git_repo / "b" / "noise.log").write_text("z\n", encoding="utf-8")
        monkeypatch.chdir(temp_git_repo / "a")

        assert sorted(list_worktree_untracked_files()) == [
            ".gitignore", "a/new.py", "b/other.txt",
        ]

    def test_outside_a_repository_it_is_unknown_not_empty(self, tmp_path, monkeypatch):
        outside = tmp_path / "plain"
        outside.mkdir()
        monkeypatch.chdir(outside)
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
        assert list_worktree_untracked_files() is None
