"""Tests for prompt loader module."""

import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.prompt_loader import (
    SKILL_DIR,
    PROMPTS_DIR,
    load_prompt,
    clear_cache,
)


class TestPromptLoaderConstants:
    """Tests for module-level constants."""

    def test_skill_dir_exists(self):
        """Test that SKILL_DIR points to a valid directory."""
        assert SKILL_DIR.exists()
        assert SKILL_DIR.is_dir()

    def test_prompts_dir_exists(self):
        """Test that PROMPTS_DIR points to a valid directory."""
        assert PROMPTS_DIR.exists()
        assert PROMPTS_DIR.is_dir()

    def test_prompts_dir_is_under_skill_dir(self):
        """Test that PROMPTS_DIR is under SKILL_DIR."""
        assert PROMPTS_DIR.parent == SKILL_DIR


class TestLoadPrompt:
    """Tests for load_prompt function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_load_existing_prompt_file(self):
        """Test loading an existing prompt file."""
        content = load_prompt("plan_review.txt")
        assert isinstance(content, str)
        assert len(content) > 0
        # Verify it contains expected content from the actual file
        assert "reviewing an implementation plan" in content

    def test_load_code_review_prompt(self):
        """Test loading code_review.txt prompt file."""
        content = load_prompt("code_review.txt")
        assert isinstance(content, str)
        assert len(content) > 0

    def test_load_implementation_task_prompt(self):
        """Test loading implementation_task.txt prompt file."""
        content = load_prompt("implementation_task.txt")
        assert isinstance(content, str)
        assert len(content) > 0

    def test_nonexistent_file_raises_file_not_found_error(self):
        """Test that loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_prompt("nonexistent_prompt.txt")
        assert "Prompt not found" in str(exc_info.value)

    def test_nonexistent_file_error_includes_path(self):
        """Test that error message includes the full path."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_prompt("does_not_exist.txt")
        error_message = str(exc_info.value)
        assert "prompts" in error_message
        assert "does_not_exist.txt" in error_message


class TestLoadPromptCaching:
    """Tests for load_prompt caching behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_same_file_loaded_twice_returns_cached(self):
        """Test that loading the same file twice uses the cache."""
        # First load
        content1 = load_prompt("plan_review.txt")
        cache_info_after_first = load_prompt.cache_info()

        # Second load
        content2 = load_prompt("plan_review.txt")
        cache_info_after_second = load_prompt.cache_info()

        # Verify same content returned
        assert content1 == content2

        # Verify cache was used (hits increased, misses stayed same)
        assert cache_info_after_second.hits == cache_info_after_first.hits + 1
        assert cache_info_after_second.misses == cache_info_after_first.misses

    def test_different_files_cached_separately(self):
        """Test that different files are cached separately."""
        content1 = load_prompt("plan_review.txt")
        content2 = load_prompt("code_review.txt")

        cache_info = load_prompt.cache_info()

        # Both should be cache misses (new files)
        assert cache_info.misses == 2
        assert cache_info.currsize == 2

        # Contents should be different
        assert content1 != content2


class TestLoadPromptWithTempFiles:
    """Tests using temporary files via monkeypatch."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_utf8_content_handled_correctly(self, temp_dir, monkeypatch):
        """Test that UTF-8 content is handled correctly."""
        # Create a temp prompt file with UTF-8 content
        utf8_content = "Hello World!\nUnicode: \u4e2d\u6587 \u65e5\u672c\u8a9e \ud55c\uad6d\uc5b4\nSpecial: \u00e9\u00e8\u00ea\u00eb\nSymbols: \u2192 \u2713 \u2717"
        temp_prompt = temp_dir / "utf8_test.txt"
        temp_prompt.write_text(utf8_content, encoding="utf-8")

        # Monkeypatch PROMPTS_DIR to use temp directory
        import utils.prompt_loader as prompt_loader_module
        monkeypatch.setattr(prompt_loader_module, "PROMPTS_DIR", temp_dir)

        # Clear cache after monkeypatching
        clear_cache()

        # Load and verify
        loaded_content = load_prompt("utf8_test.txt")
        assert loaded_content == utf8_content

    def test_multiline_content_preserved(self, temp_dir, monkeypatch):
        """Test that multiline content is preserved correctly."""
        multiline_content = """Line 1
Line 2
Line 3

Line 5 after blank line
    Indented line
\tTab indented line"""
        temp_prompt = temp_dir / "multiline_test.txt"
        temp_prompt.write_text(multiline_content, encoding="utf-8")

        import utils.prompt_loader as prompt_loader_module
        monkeypatch.setattr(prompt_loader_module, "PROMPTS_DIR", temp_dir)
        clear_cache()

        loaded_content = load_prompt("multiline_test.txt")
        assert loaded_content == multiline_content

    def test_empty_file_loads_correctly(self, temp_dir, monkeypatch):
        """Test that empty files load correctly."""
        temp_prompt = temp_dir / "empty.txt"
        temp_prompt.write_text("", encoding="utf-8")

        import utils.prompt_loader as prompt_loader_module
        monkeypatch.setattr(prompt_loader_module, "PROMPTS_DIR", temp_dir)
        clear_cache()

        loaded_content = load_prompt("empty.txt")
        assert loaded_content == ""


class TestClearCache:
    """Tests for clear_cache function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_clear_cache_resets_cache_info(self):
        """Test that clear_cache resets the cache."""
        # Load a file to populate cache
        load_prompt("plan_review.txt")
        cache_info_before = load_prompt.cache_info()
        assert cache_info_before.currsize > 0

        # Clear cache
        clear_cache()

        # Verify cache is empty
        cache_info_after = load_prompt.cache_info()
        assert cache_info_after.currsize == 0
        assert cache_info_after.hits == 0
        assert cache_info_after.misses == 0

    def test_after_clear_cache_file_is_read_from_disk(self, temp_dir, monkeypatch):
        """Test that after clearing cache, load_prompt reads from disk again."""
        # Create a temp file
        initial_content = "Initial content"
        temp_prompt = temp_dir / "mutable_test.txt"
        temp_prompt.write_text(initial_content, encoding="utf-8")

        import utils.prompt_loader as prompt_loader_module
        monkeypatch.setattr(prompt_loader_module, "PROMPTS_DIR", temp_dir)
        clear_cache()

        # First load
        content1 = load_prompt("mutable_test.txt")
        assert content1 == initial_content

        # Modify the file on disk
        updated_content = "Updated content"
        temp_prompt.write_text(updated_content, encoding="utf-8")

        # Load again without clearing - should get cached version
        content2 = load_prompt("mutable_test.txt")
        assert content2 == initial_content  # Still cached

        # Clear cache and load again - should get new content
        clear_cache()
        content3 = load_prompt("mutable_test.txt")
        assert content3 == updated_content  # Read from disk

    def test_clear_cache_returns_none(self):
        """Test that clear_cache returns None."""
        result = clear_cache()
        assert result is None


class TestLoadPromptEdgeCases:
    """Tests for edge cases in load_prompt."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_prompt_name_with_subdirectory_path_traversal(self, temp_dir, monkeypatch):
        """Test handling of path with directory traversal."""
        # This should still work as a relative path within prompts dir
        import utils.prompt_loader as prompt_loader_module
        monkeypatch.setattr(prompt_loader_module, "PROMPTS_DIR", temp_dir)

        # Create a subdirectory with a file
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        subfile = subdir / "nested.txt"
        subfile.write_text("Nested content", encoding="utf-8")

        clear_cache()

        # Should be able to load with subdirectory path
        content = load_prompt("subdir/nested.txt")
        assert content == "Nested content"

    def test_load_prompt_with_special_characters_in_content(self, temp_dir, monkeypatch):
        """Test loading prompt with special characters."""
        special_content = "Template: {variable}\nBackslash: \\\nQuotes: \"'`\nAngle: <>"
        temp_prompt = temp_dir / "special.txt"
        temp_prompt.write_text(special_content, encoding="utf-8")

        import utils.prompt_loader as prompt_loader_module
        monkeypatch.setattr(prompt_loader_module, "PROMPTS_DIR", temp_dir)
        clear_cache()

        loaded_content = load_prompt("special.txt")
        assert loaded_content == special_content
