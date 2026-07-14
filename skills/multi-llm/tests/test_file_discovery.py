"""Tests for file_discovery module."""

import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.file_discovery import FileDiscovery, discover_implementation_context


class TestFileDiscoveryInit:
    """Tests for FileDiscovery.__init__() method."""

    def test_default_ignore_patterns_applied(self, tmp_path):
        """Test that default ignore patterns are applied."""
        discovery = FileDiscovery(tmp_path)

        assert discovery.ignore_patterns == FileDiscovery.DEFAULT_IGNORE_PATTERNS
        assert "*.pyc" in discovery.ignore_patterns
        assert "__pycache__" in discovery.ignore_patterns
        assert ".git" in discovery.ignore_patterns

    def test_custom_patterns_added(self, tmp_path):
        """Test that custom ignore patterns are added to defaults."""
        custom_patterns = ["*.tmp", "*.bak"]
        discovery = FileDiscovery(tmp_path, ignore_patterns=custom_patterns)

        assert "*.tmp" in discovery.ignore_patterns
        assert "*.bak" in discovery.ignore_patterns
        # Default patterns should still be present
        assert "*.pyc" in discovery.ignore_patterns
        assert "__pycache__" in discovery.ignore_patterns

    def test_root_path_resolved(self, tmp_path):
        """Test that root_path is resolved to absolute path."""
        discovery = FileDiscovery(tmp_path)

        assert discovery.root_path.is_absolute()
        assert discovery.root_path == tmp_path.resolve()

    def test_custom_patterns_none_uses_defaults(self, tmp_path):
        """Test that None custom patterns uses only defaults."""
        discovery = FileDiscovery(tmp_path, ignore_patterns=None)

        assert discovery.ignore_patterns == FileDiscovery.DEFAULT_IGNORE_PATTERNS

    def test_custom_patterns_empty_list_uses_defaults(self, tmp_path):
        """Test that empty custom patterns list uses only defaults."""
        discovery = FileDiscovery(tmp_path, ignore_patterns=[])

        assert discovery.ignore_patterns == FileDiscovery.DEFAULT_IGNORE_PATTERNS


class TestShouldIgnore:
    """Tests for FileDiscovery._should_ignore() method."""

    def test_matches_pyc_extension(self, tmp_path):
        """Test that .pyc files are ignored."""
        discovery = FileDiscovery(tmp_path)
        pyc_file = tmp_path / "module.pyc"
        pyc_file.touch()

        assert discovery._should_ignore(pyc_file) is True

    def test_matches_pycache_directory(self, tmp_path):
        """Test that __pycache__ directories are ignored."""
        discovery = FileDiscovery(tmp_path)
        pycache_dir = tmp_path / "__pycache__"
        pycache_dir.mkdir()

        assert discovery._should_ignore(pycache_dir) is True

    def test_matches_git_directory(self, tmp_path):
        """Test that .git directory is ignored."""
        discovery = FileDiscovery(tmp_path)
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        assert discovery._should_ignore(git_dir) is True

    def test_matches_custom_patterns(self, tmp_path):
        """Test that custom patterns are matched."""
        discovery = FileDiscovery(tmp_path, ignore_patterns=["*.custom", "secret_*"])
        custom_file = tmp_path / "file.custom"
        custom_file.touch()
        secret_file = tmp_path / "secret_data.txt"
        secret_file.touch()

        assert discovery._should_ignore(custom_file) is True
        assert discovery._should_ignore(secret_file) is True

    def test_does_not_match_regular_file(self, tmp_path):
        """Test that regular files are not ignored."""
        discovery = FileDiscovery(tmp_path)
        regular_file = tmp_path / "main.py"
        regular_file.touch()

        assert discovery._should_ignore(regular_file) is False

    def test_matches_node_modules(self, tmp_path):
        """Test that node_modules directory is ignored."""
        discovery = FileDiscovery(tmp_path)
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()

        assert discovery._should_ignore(node_modules) is True

    def test_matches_log_files(self, tmp_path):
        """Test that .log files are ignored."""
        discovery = FileDiscovery(tmp_path)
        log_file = tmp_path / "app.log"
        log_file.touch()

        assert discovery._should_ignore(log_file) is True

    def test_matches_venv_directory(self, tmp_path):
        """Test that venv directories are ignored."""
        discovery = FileDiscovery(tmp_path)
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        dot_venv_dir = tmp_path / ".venv"
        dot_venv_dir.mkdir()

        assert discovery._should_ignore(venv_dir) is True
        assert discovery._should_ignore(dot_venv_dir) is True


class TestFindByExtension:
    """Tests for FileDiscovery.find_by_extension() method."""

    def test_finds_py_files(self, tmp_path):
        """Test finding .py files."""
        discovery = FileDiscovery(tmp_path)
        # Create test files
        (tmp_path / "main.py").touch()
        (tmp_path / "utils.py").touch()
        (tmp_path / "config.json").touch()

        results = discovery.find_by_extension([".py"])

        assert len(results) == 2
        assert all(f.suffix == ".py" for f in results)

    def test_finds_multiple_extensions(self, tmp_path):
        """Test finding files with multiple extensions."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "main.py").touch()
        (tmp_path / "script.js").touch()
        (tmp_path / "style.css").touch()
        (tmp_path / "config.yaml").touch()

        results = discovery.find_by_extension([".py", ".js"])

        assert len(results) == 2
        extensions = {f.suffix for f in results}
        assert extensions == {".py", ".js"}

    def test_ignores_excluded_dirs(self, tmp_path):
        """Test that files in excluded directories are ignored."""
        discovery = FileDiscovery(tmp_path)
        # Create files in normal directory
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        # Create files in ignored directory
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "package.py").touch()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.py").touch()

        results = discovery.find_by_extension([".py"])

        assert len(results) == 1
        assert results[0].name == "main.py"

    def test_finds_files_in_subdirectories(self, tmp_path):
        """Test finding files in nested subdirectories."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "core").mkdir()
        (tmp_path / "main.py").touch()
        (tmp_path / "src" / "utils.py").touch()
        (tmp_path / "src" / "core" / "engine.py").touch()

        results = discovery.find_by_extension([".py"])

        assert len(results) == 3

    def test_returns_sorted_results(self, tmp_path):
        """Test that results are sorted."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "z_module.py").touch()
        (tmp_path / "a_module.py").touch()
        (tmp_path / "m_module.py").touch()

        results = discovery.find_by_extension([".py"])

        assert results == sorted(results)

    def test_empty_results_for_no_matches(self, tmp_path):
        """Test empty results when no files match."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "file.txt").touch()

        results = discovery.find_by_extension([".py"])

        assert results == []


class TestFindByNamePattern:
    """Tests for FileDiscovery.find_by_name_pattern() method."""

    def test_matches_test_pattern(self, tmp_path):
        """Test matching test_*.py pattern."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "test_main.py").touch()
        (tmp_path / "test_utils.py").touch()
        (tmp_path / "main.py").touch()

        results = discovery.find_by_name_pattern("test_*.py")

        assert len(results) == 2
        assert all(f.name.startswith("test_") for f in results)

    def test_matches_yaml_pattern(self, tmp_path):
        """Test matching *.yaml pattern."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "config.yaml").touch()
        (tmp_path / "settings.yaml").touch()
        (tmp_path / "config.yml").touch()
        (tmp_path / "config.json").touch()

        results = discovery.find_by_name_pattern("*.yaml")

        assert len(results) == 2
        assert all(f.suffix == ".yaml" for f in results)

    def test_matches_yml_pattern(self, tmp_path):
        """Test matching *.yml pattern."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "config.yml").touch()
        (tmp_path / "docker-compose.yml").touch()

        results = discovery.find_by_name_pattern("*.yml")

        assert len(results) == 2

    def test_ignores_excluded_directories(self, tmp_path):
        """Test that pattern matching ignores excluded directories."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "test_main.py").touch()
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "test_config.py").touch()

        results = discovery.find_by_name_pattern("test_*.py")

        assert len(results) == 1
        assert results[0].name == "test_main.py"

    def test_matches_conftest_pattern(self, tmp_path):
        """Test matching conftest*.py pattern."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "conftest.py").touch()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "conftest.py").touch()

        results = discovery.find_by_name_pattern("conftest*.py")

        assert len(results) == 2

    def test_matches_init_pattern(self, tmp_path):
        """Test matching __init__.py pattern."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "__init__.py").touch()
        (tmp_path / "module").mkdir()
        (tmp_path / "module" / "__init__.py").touch()

        results = discovery.find_by_name_pattern("__init__.py")

        assert len(results) == 2


class TestFindByContent:
    """Tests for FileDiscovery.find_by_content() method."""

    def test_plain_text_search(self, tmp_path):
        """Test plain text content search."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "file1.py").write_text("def hello_world():\n    pass", encoding="utf-8")
        (tmp_path / "file2.py").write_text("def goodbye():\n    pass", encoding="utf-8")
        (tmp_path / "file3.py").write_text("# hello world comment", encoding="utf-8")

        results = discovery.find_by_content("hello")

        assert len(results) == 2
        file_names = {r[0].name for r in results}
        assert file_names == {"file1.py", "file3.py"}

    def test_regex_search(self, tmp_path):
        """Test regex content search."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "file1.py").write_text("def process_data():\n    pass", encoding="utf-8")
        (tmp_path / "file2.py").write_text("def process_items():\n    pass", encoding="utf-8")
        (tmp_path / "file3.py").write_text("def handle_data():\n    pass", encoding="utf-8")

        results = discovery.find_by_content(r"def process_\w+\(\)", regex=True)

        assert len(results) == 2
        file_names = {r[0].name for r in results}
        assert file_names == {"file1.py", "file2.py"}

    def test_limits_to_extensions(self, tmp_path):
        """Test that search is limited to specified extensions."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "code.py").write_text("hello world", encoding="utf-8")
        (tmp_path / "doc.md").write_text("hello world", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("hello: world", encoding="utf-8")

        results = discovery.find_by_content("hello", extensions=[".py"])

        assert len(results) == 1
        assert results[0][0].name == "code.py"

    def test_returns_line_numbers(self, tmp_path):
        """Test that matching line numbers are returned."""
        discovery = FileDiscovery(tmp_path)
        content = "line 1\nfind me\nline 3\nfind me again\nline 5"
        (tmp_path / "file.txt").write_text(content, encoding="utf-8")

        results = discovery.find_by_content("find me")

        assert len(results) == 1
        file_path, line_numbers = results[0]
        assert line_numbers == [2, 4]

    def test_handles_binary_files_gracefully(self, tmp_path):
        """Test that binary files are handled without crashing."""
        discovery = FileDiscovery(tmp_path)
        binary_content = bytes([0x00, 0x01, 0x02, 0xFF, 0xFE])
        (tmp_path / "binary.dat").write_bytes(binary_content)
        (tmp_path / "text.txt").write_text("searchable text", encoding="utf-8")

        # Should not raise exception
        results = discovery.find_by_content("searchable")

        assert len(results) == 1
        assert results[0][0].name == "text.txt"

    def test_case_sensitive_search(self, tmp_path):
        """Test that search is case-sensitive by default."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "file1.py").write_text("HELLO", encoding="utf-8")
        (tmp_path / "file2.py").write_text("hello", encoding="utf-8")
        (tmp_path / "file3.py").write_text("Hello", encoding="utf-8")

        results = discovery.find_by_content("hello")

        assert len(results) == 1
        assert results[0][0].name == "file2.py"

    def test_regex_case_insensitive(self, tmp_path):
        """Test case-insensitive regex search."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "file1.py").write_text("HELLO", encoding="utf-8")
        (tmp_path / "file2.py").write_text("hello", encoding="utf-8")

        results = discovery.find_by_content(r"(?i)hello", regex=True)

        assert len(results) == 2


class TestAllFiles:
    """Tests for FileDiscovery._all_files() method."""

    def test_returns_all_non_ignored_files(self, tmp_path):
        """Test that all non-ignored files are returned."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "file1.py").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "file3.md").touch()

        results = discovery._all_files()

        assert len(results) == 3

    def test_excludes_ignored_files(self, tmp_path):
        """Test that ignored files are excluded."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "main.py").touch()
        (tmp_path / "main.pyc").touch()
        (tmp_path / "app.log").touch()

        results = discovery._all_files()

        assert len(results) == 1
        assert results[0].name == "main.py"

    def test_excludes_files_in_ignored_directories(self, tmp_path):
        """Test that files in ignored directories are excluded."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-39.pyc").touch()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "package" / "index.js").parents[0].mkdir(parents=True)
        (tmp_path / "node_modules" / "package" / "index.js").touch()

        results = discovery._all_files()

        assert len(results) == 1
        assert results[0].name == "main.py"

    def test_traverses_subdirectories(self, tmp_path):
        """Test that subdirectories are traversed."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "level2").mkdir()
        (tmp_path / "file1.py").touch()
        (tmp_path / "level1" / "file2.py").touch()
        (tmp_path / "level1" / "level2" / "file3.py").touch()

        results = discovery._all_files()

        assert len(results) == 3


class TestFindRelatedFiles:
    """Tests for FileDiscovery.find_related_files() method."""

    def test_finds_test_files(self, tmp_path):
        """Test finding test files for a source file."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "utils.py").write_text("def helper(): pass", encoding="utf-8")
        (tmp_path / "test_utils.py").touch()
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_utils.py").touch()

        results = discovery.find_related_files(tmp_path / "utils.py")

        assert len(results["tests"]) >= 1
        test_names = {f.name for f in results["tests"]}
        assert "test_utils.py" in test_names

    def test_finds_same_directory_files(self, tmp_path):
        """Test finding files in the same directory."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("import utils", encoding="utf-8")
        (tmp_path / "src" / "utils.py").write_text("def helper(): pass", encoding="utf-8")
        (tmp_path / "src" / "config.py").write_text("CONFIG = {}", encoding="utf-8")

        results = discovery.find_related_files(tmp_path / "src" / "main.py")

        assert len(results["same_directory"]) == 2
        sibling_names = {f.name for f in results["same_directory"]}
        assert "utils.py" in sibling_names
        assert "config.py" in sibling_names

    def test_handles_missing_file(self, tmp_path):
        """Test handling of non-existent file."""
        discovery = FileDiscovery(tmp_path)

        results = discovery.find_related_files(tmp_path / "nonexistent.py")

        assert results["tests"] == []
        assert results["imports"] == []
        assert results["imported_by"] == []
        assert results["same_directory"] == []

    def test_finds_files_that_import_target(self, tmp_path):
        """Test finding files that import the target file."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "utils.py").write_text("def helper(): pass", encoding="utf-8")
        (tmp_path / "main.py").write_text("from utils import helper\nhelper()", encoding="utf-8")
        (tmp_path / "other.py").write_text("import utils\nutils.helper()", encoding="utf-8")

        results = discovery.find_related_files(tmp_path / "utils.py")

        assert len(results["imported_by"]) >= 1

    def test_finds_imported_modules(self, tmp_path):
        """Test finding modules imported by the target file."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "helpers.py").write_text("def helper(): pass", encoding="utf-8")
        (tmp_path / "main.py").write_text("import helpers\nfrom helpers import helper", encoding="utf-8")

        results = discovery.find_related_files(tmp_path / "main.py")

        # Should find helpers.py as an import
        import_names = {f.name for f in results["imports"]}
        assert "helpers.py" in import_names

    def test_skips_tests_for_test_files(self, tmp_path):
        """Test that test files don't search for their own tests."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "test_utils.py").write_text("def test_something(): pass", encoding="utf-8")

        results = discovery.find_related_files(tmp_path / "test_utils.py")

        # Test files (starting with test_) should not have tests looked up
        assert len(results["tests"]) == 0

    def test_respects_include_tests_flag(self, tmp_path):
        """Test that include_tests=False skips test discovery."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "utils.py").write_text("def helper(): pass", encoding="utf-8")
        (tmp_path / "test_utils.py").touch()

        results = discovery.find_related_files(tmp_path / "utils.py", include_tests=False)

        assert results["tests"] == []

    def test_respects_include_imports_flag(self, tmp_path):
        """Test that include_imports=False skips import discovery."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "helpers.py").write_text("def helper(): pass", encoding="utf-8")
        (tmp_path / "main.py").write_text("import helpers", encoding="utf-8")

        results = discovery.find_related_files(tmp_path / "main.py", include_imports=False)

        assert results["imports"] == []
        assert results["imported_by"] == []

    def test_only_includes_same_extension_siblings(self, tmp_path):
        """Test that only same extension files are in same_directory."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "main.py").write_text("pass", encoding="utf-8")
        (tmp_path / "config.py").write_text("pass", encoding="utf-8")
        (tmp_path / "readme.md").write_text("# Readme", encoding="utf-8")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")

        results = discovery.find_related_files(tmp_path / "main.py")

        sibling_names = {f.name for f in results["same_directory"]}
        assert "config.py" in sibling_names
        assert "readme.md" not in sibling_names
        assert "data.json" not in sibling_names


class TestGetProjectStructure:
    """Tests for FileDiscovery.get_project_structure() method."""

    def test_returns_structure_dict(self, tmp_path):
        """Test that a structure dictionary is returned."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "file.py").touch()

        structure = discovery.get_project_structure()

        assert isinstance(structure, dict)
        assert "root" in structure
        assert "directories" in structure
        assert "file_counts" in structure
        assert "total_files" in structure

    def test_file_counts_by_extension(self, tmp_path):
        """Test that file counts are grouped by extension."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "file1.py").touch()
        (tmp_path / "file2.py").touch()
        (tmp_path / "config.yaml").touch()
        (tmp_path / "readme.md").touch()

        structure = discovery.get_project_structure()

        assert structure["file_counts"][".py"] == 2
        assert structure["file_counts"][".yaml"] == 1
        assert structure["file_counts"][".md"] == 1
        assert structure["total_files"] == 4

    def test_respects_max_depth(self, tmp_path):
        """Test that max_depth limits directory exploration."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "level2").mkdir()
        (tmp_path / "level1" / "level2" / "level3").mkdir()
        (tmp_path / "level1" / "level2" / "level3" / "level4").mkdir()
        (tmp_path / "level1" / "file1.py").touch()
        (tmp_path / "level1" / "level2" / "file2.py").touch()
        (tmp_path / "level1" / "level2" / "level3" / "file3.py").touch()
        (tmp_path / "level1" / "level2" / "level3" / "level4" / "file4.py").touch()

        structure = discovery.get_project_structure(max_depth=2)

        # Files at depth > max_depth should not be counted
        # level1 is depth 1, level2 is depth 2, level3 is depth 3
        assert structure["total_files"] == 2  # file1.py and file2.py

    def test_excludes_ignored_directories(self, tmp_path):
        """Test that ignored directories are excluded from structure."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").touch()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "package.json").touch()

        structure = discovery.get_project_structure()

        dir_names = [d["name"] for d in structure["directories"]]
        assert "src" in dir_names
        assert "node_modules" not in dir_names
        assert structure["total_files"] == 1

    def test_root_path_in_structure(self, tmp_path):
        """Test that root path is included in structure."""
        discovery = FileDiscovery(tmp_path)

        structure = discovery.get_project_structure()

        assert structure["root"] == str(tmp_path.resolve())

    def test_handles_empty_directory(self, tmp_path):
        """Test handling of empty directory."""
        discovery = FileDiscovery(tmp_path)

        structure = discovery.get_project_structure()

        assert structure["total_files"] == 0
        assert structure["file_counts"] == {}

    def test_counts_files_without_extension(self, tmp_path):
        """Test that files without extension are counted."""
        discovery = FileDiscovery(tmp_path)
        (tmp_path / "Makefile").touch()
        (tmp_path / "Dockerfile").touch()
        (tmp_path / "script.py").touch()

        structure = discovery.get_project_structure()

        assert structure["file_counts"]["no_extension"] == 2
        assert structure["file_counts"][".py"] == 1
        assert structure["total_files"] == 3


class TestDiscoverImplementationContext:
    """Tests for discover_implementation_context() function."""

    def test_extracts_file_names_from_task_description(self, tmp_path):
        """Test extracting file names from task description."""
        (tmp_path / "config.yaml").write_text("key: value", encoding="utf-8")
        (tmp_path / "settings.yaml").touch()

        results = discover_implementation_context(
            tmp_path,
            'Update the `config.yaml` file to add new settings'
        )

        mentioned_names = {f.name for f in results["mentioned_files"]}
        assert "config.yaml" in mentioned_names

    def test_extracts_module_names_from_description(self, tmp_path):
        """Test extracting module/class names from task description."""
        (tmp_path / "user_service.py").write_text("class UserService: pass", encoding="utf-8")
        (tmp_path / "other.py").write_text("# unrelated", encoding="utf-8")

        results = discover_implementation_context(
            tmp_path,
            'Modify the `UserService` class to add authentication'
        )

        python_names = {f.name for f in results["related_python"]}
        assert "user_service.py" in python_names

    def test_finds_config_files(self, tmp_path):
        """Test finding configuration files."""
        (tmp_path / "config.yaml").touch()
        (tmp_path / "settings.yml").touch()
        (tmp_path / "package.json").touch()
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "setup.cfg").touch()

        results = discover_implementation_context(tmp_path, "Add a new feature")

        config_names = {f.name for f in results["config_files"]}
        assert "config.yaml" in config_names
        assert "settings.yml" in config_names
        assert "package.json" in config_names
        assert "pyproject.toml" in config_names
        assert "setup.cfg" in config_names

    def test_deduplicates_results(self, tmp_path):
        """Test that results are deduplicated."""
        (tmp_path / "config.yaml").write_text("config: true", encoding="utf-8")

        results = discover_implementation_context(
            tmp_path,
            'Check `config.yaml` and also look at "config.yaml"'
        )

        # config.yaml should only appear once in mentioned_files
        mentioned = results["mentioned_files"]
        assert len([f for f in mentioned if f.name == "config.yaml"]) == 1

    def test_returns_empty_for_no_matches(self, tmp_path):
        """Test empty results when nothing matches."""
        (tmp_path / "unrelated.txt").touch()

        results = discover_implementation_context(
            tmp_path,
            "Do something generic"
        )

        assert results["mentioned_files"] == []
        assert results["related_python"] == []

    def test_handles_backtick_and_quote_patterns(self, tmp_path):
        """Test extraction from both backticks and quotes."""
        # The files must contain the search term since find_by_content is used
        (tmp_path / "module_a.py").write_text("module_a implementation", encoding="utf-8")
        (tmp_path / "module_b.py").write_text("module_b implementation", encoding="utf-8")

        results = discover_implementation_context(
            tmp_path,
            'Update `module_a` and also check "module_b" for issues'
        )

        python_names = {f.name for f in results["related_python"]}
        assert "module_a.py" in python_names
        assert "module_b.py" in python_names

    def test_ignores_excluded_directories_in_context(self, tmp_path):
        """Test that excluded directories are ignored in context discovery."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "config.yaml").touch()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "config.yaml").touch()

        results = discover_implementation_context(tmp_path, "Check configs")

        config_paths = [str(f) for f in results["config_files"]]
        assert any("src" in p for p in config_paths)
        assert not any("node_modules" in p for p in config_paths)
