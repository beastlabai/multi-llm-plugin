#!/usr/bin/env python3
"""Integration tests for code review orchestrator."""

import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_review_orchestrator import format_tracked_files_for_prompt
from utils.schema_validator import validate_code_review_issues
from utils.suggestion_processor import group_similar_suggestions


class TestCodeReviewIssueGrouping:
    """Tests for code review issue grouping."""

    def test_group_code_review_issues(self):
        """Test grouping of code review issues."""
        issues = [
            {
                "title": "Unused variable",
                "desc": "Variable 'x' is declared but never used",
                "importance": "low",
                "file": "src/main.py",
                "type": "style",
                "source_model": "gpt-4"
            },
            {
                "title": "Unused variable found",
                "desc": "The variable 'x' in main.py is not used",
                "importance": "low",
                "file": "src/main.py",
                "type": "style",
                "source_model": "claude"
            }
        ]
        result = group_similar_suggestions(issues)
        # Similar issues should potentially be grouped
        assert len(result) >= 1

    def test_different_file_issues_not_merged(self):
        """Issues in different files stay separate.

        A code-review finding is anchored to a file and its fix lands in one,
        so two findings naming different files are two fixes — word-identical
        or not. The score halves when the file anchors disagree, which is what
        keeps an identical-sounding finding in ``a.py`` and ``b.py`` apart.
        """
        issues = [
            {
                "title": "Missing null check",
                "desc": "Add null check",
                "importance": "high",
                "file": "src/a.py",
                "type": "bug",
                "source_model": "gpt-4"
            },
            {
                "title": "Missing null check",
                "desc": "Add null check",
                "importance": "high",
                "file": "src/b.py",
                "type": "bug",
                "source_model": "claude"
            }
        ]
        result = group_similar_suggestions(issues)

        assert len(result) == 2

    def test_same_file_issues_still_merge(self):
        """The file rule only ever separates — same file, same finding, one group."""
        issues = [
            {
                "title": "Missing null check on the session lookup",
                "desc": "get_session dereferences the row without checking for None",
                "importance": "high",
                "file": "src/a.py:40-52",
                "type": "bug",
                "source_model": "gpt-4"
            },
            {
                "title": "Session lookup missing a null check",
                "desc": "The session row is dereferenced with no None check in get_session",
                "importance": "high",
                "file": "a.py",
                "type": "bug",
                "source_model": "claude"
            }
        ]
        result = group_similar_suggestions(issues)

        # The anchors differ only in path spelling and line range, so they
        # reduce to the same file and the pair is free to merge.
        assert len(result) == 1
        assert len(result[0].suggestions) == 2

    def test_same_basename_in_different_directories_stays_apart(self):
        """A suffix match must land on a path boundary, or a monorepo merges itself."""
        issues = [
            {
                "title": "Missing null check on the session lookup",
                "desc": "get_session dereferences the row without checking for None",
                "importance": "high",
                "file": "services/a/utils.py",
                "type": "bug",
                "source_model": "gpt-4"
            },
            {
                "title": "Missing null check on the session lookup",
                "desc": "get_session dereferences the row without checking for None",
                "importance": "high",
                "file": "services/b/utils.py",
                "type": "bug",
                "source_model": "claude"
            }
        ]
        result = group_similar_suggestions(issues)

        assert len(result) == 2


class TestCodeReviewValidation:
    """Tests for code review issue validation."""

    def test_validate_valid_issues(self):
        """Test validation of properly formatted issues."""
        issues = [
            {
                "title": "Test issue",
                "desc": "Test description",
                "importance": "high",
                "file": "test.py",
                "type": "bug"
            }
        ]
        is_valid, errors = validate_code_review_issues(issues)
        assert is_valid
        assert not errors

    def test_validate_missing_fields(self):
        """Test validation catches missing fields."""
        issues = [
            {
                "title": "Test issue"
                # Missing other required fields
            }
        ]
        is_valid, errors = validate_code_review_issues(issues)
        # Validation should catch missing fields
        assert isinstance(errors, list)


class TestFormatTrackedFilesForPrompt:
    """Tests for the format_tracked_files_for_prompt function."""

    def test_empty_tracked_files(self):
        """An empty scope says so in one line; the old placeholder is gone.

        The placeholder "(No tracked files - using git diff to discover
        changes)" reached every reviewer of a 1 040-file fallback scope in
        place of the list (review-scope plan §2.1). The prompt's scope-source
        line now says why a list is empty.
        """
        from code_review_orchestrator import NO_CHANGED_FILES_LINE

        result = format_tracked_files_for_prompt([])
        assert result == NO_CHANGED_FILES_LINE
        assert "no changed files found" in result.lower()
        assert "No tracked files - using git diff" not in result

    def test_normal_list(self):
        """A small list is returned fully, no truncation, with task attribution."""
        paths = ["src/a.py", "src/b.py", "src/c.py"]
        task_ids = {"src/a.py": "task-1", "src/b.py": "task-2", "src/c.py": "task-3"}
        result = format_tracked_files_for_prompt(paths, task_ids=task_ids)
        assert "src/a.py" in result
        assert "src/b.py" in result
        assert "src/c.py" in result
        assert "truncated" not in result
        assert result.splitlines() == [
            "- `src/a.py` (from task: task-1)",
            "- `src/b.py` (from task: task-2)",
            "- `src/c.py` (from task: task-3)",
        ]

    def test_a_scope_without_task_ids_lists_bare_paths(self):
        """A git-diff or --files path has no task, so no `(from task: …)` at all
        — never `(from task: unknown)`."""
        result = format_tracked_files_for_prompt(["src/a.py", "src/b.py"])
        assert result.splitlines() == ["- `src/a.py`", "- `src/b.py`"]
        assert "(from task:" not in result

    def test_the_given_order_is_kept(self):
        """The caller's review order survives: source first, tests after."""
        result = format_tracked_files_for_prompt(["utils/z.py", "tests/test_a.py"])
        assert result.index("utils/z.py") < result.index("tests/test_a.py")

    def test_truncation_with_many_files(self):
        """200 tracked files should be truncated at the default max_chars=5000."""
        paths = [f"src/file_{i}.py" for i in range(200)]
        # The same entries this test always used (every path from `task-1`):
        # bare, 200 short paths fit under 5 000 characters.
        result = format_tracked_files_for_prompt(
            paths, task_ids={path: "task-1" for path in paths}
        )
        assert "more files not shown" in result
        assert "truncated to stay within context limits" in result
        # The text before the notice should be shorter than 5100 chars (some tolerance)
        notice_idx = result.index("truncated to stay within context limits")
        text_before = result[:notice_idx]
        assert len(text_before) < 5100

    def test_truncation_with_custom_threshold(self):
        """20 entries with max_chars=100 should trigger truncation."""
        paths = [f"src/file_{i}.py" for i in range(20)]
        result = format_tracked_files_for_prompt(paths, max_chars=100)
        assert "truncated to stay within context limits" in result
        assert "more files not shown" in result

    def test_no_truncation_when_under_threshold(self):
        """5 entries with default max_chars should not be truncated."""
        paths = [f"src/file_{i}.py" for i in range(5)]
        result = format_tracked_files_for_prompt(paths)
        assert "truncated" not in result

    def test_a_cut_list_names_the_full_scope_when_given_one(self):
        """The notice stays word for word; one sentence after it names where the
        whole list is, so the cut paths are not read as out of scope."""
        paths = [f"src/file_{i}.py" for i in range(20)]
        result = format_tracked_files_for_prompt(
            paths, max_chars=100, full_scope_ref="/abs/code-review/review-scope.md"
        )
        assert (
            "more files not shown — list truncated to stay within context limits ...]"
            in result
        )
        assert result.endswith(
            "\nEvery path not shown is still in scope; the full list is "
            "/abs/code-review/review-scope.md."
        )

    def test_an_uncut_list_carries_no_full_scope_sentence(self):
        result = format_tracked_files_for_prompt(
            ["src/a.py"], full_scope_ref="/abs/code-review/review-scope.md"
        )
        assert "Every path not shown" not in result
        assert result == "- `src/a.py`"

    def test_ordered_fixtures_cannot_push_source_out_of_a_cut_list(self):
        """With many fixtures, `order_review_paths` then the formatter still keeps
        `utils/` in the truncated list — in plain path order `…/tests/…` sorts
        first and the cut dropped every source file (review-scope plan §2.1)."""
        from utils.git_utils import order_review_paths

        fixtures = [
            f"skills/multi-llm/tests/fixtures/opencode/case_{i:04d}.json"
            for i in range(900)
        ]
        source = ["skills/multi-llm/utils/git_utils.py", "skills/multi-llm/utils/zz.py"]
        # Path order alone would cut the source: prove the premise first.
        assert "utils/git_utils.py" not in format_tracked_files_for_prompt(
            sorted(fixtures + source)
        )
        result = format_tracked_files_for_prompt(order_review_paths(fixtures + source))
        assert "more files not shown" in result
        assert "skills/multi-llm/utils/git_utils.py" in result
        assert "skills/multi-llm/utils/zz.py" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
