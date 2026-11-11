"""``tests/source_cache``: the shared parsed-source cache stays fresh within one process.

The contract is in ``tests/source_cache.py``'s module docstring: directory
membership is rescanned on every call, and a cached tree or derived value is
reused only while the file's bytes are unchanged, whatever its timestamp says.
"""
import ast
import gc
import os
import weakref

from tests.source_cache import derived, python_files, source_text, source_tree


def _assigned_name(tree):
    return tree.body[0].targets[0].id


class TestDirectoryMembership:
    def test_a_module_added_between_two_walks_is_listed(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        assert [p.name for p in python_files(tmp_path)] == ["a.py"]

        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "bad.py").write_text("import yaml\n", encoding="utf-8")

        assert [p.relative_to(tmp_path).as_posix() for p in python_files(tmp_path)] == [
            "a.py", "pkg/bad.py"]

    def test_a_module_deleted_between_two_walks_is_dropped(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "gone.py").write_text("y = 2\n", encoding="utf-8")
        assert [p.name for p in python_files(tmp_path)] == ["a.py", "gone.py"]

        (tmp_path / "gone.py").unlink()

        assert [p.name for p in python_files(tmp_path)] == ["a.py"]


class TestContentFreshness:
    def test_an_equal_length_rewrite_with_its_timestamp_restored_is_reread(self, tmp_path):
        module = tmp_path / "m.py"
        module.write_text("safe = 1\n", encoding="utf-8")
        before = os.stat(module)
        assert _assigned_name(source_tree(module)) == "safe"
        assert source_text(module) == "safe = 1\n"

        module.write_text("evil = 2\n", encoding="utf-8")
        os.utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = os.stat(module)
        assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)

        assert source_text(module) == "evil = 2\n"
        assert _assigned_name(source_tree(module)) == "evil"

    def test_unchanged_content_is_parsed_once(self, tmp_path):
        """The cache still pays: a second call returns the very same tree."""
        module = tmp_path / "m.py"
        module.write_text("safe = 1\n", encoding="utf-8")
        first = source_tree(module)
        os.utime(module, ns=(0, 0))  # a timestamp change alone is not a new version

        assert source_tree(module) is first
        assert isinstance(first, ast.Module)

    def test_the_tree_is_parsed_from_the_text_source_text_returns(self, tmp_path):
        """Both read the file the same way (UTF-8, universal newlines)."""
        module = tmp_path / "crlf.py"
        module.write_bytes("name = 'é'\r\nother = 2\r\n".encode("utf-8"))

        assert source_text(module) == "name = 'é'\nother = 2\n"
        tree = source_tree(module)
        assert ast.get_source_segment(source_text(module), tree.body[1]) == "other = 2"


_SEEN = []


def _record(tree):
    """A derived function that notes each tree it is handed (weakly)."""
    _SEEN.append(weakref.ref(tree))
    return _assigned_name(tree)


class TestDerivedValues:
    def setup_method(self):
        _SEEN.clear()

    def test_unchanged_content_computes_once(self, tmp_path):
        module = tmp_path / "m.py"
        module.write_text("safe = 1\n", encoding="utf-8")
        assert derived(module, _record) == "safe"
        os.utime(module, ns=(0, 0))  # a timestamp change alone is not a new version

        assert derived(module, _record) == "safe"
        assert len(_SEEN) == 1

    def test_an_equal_length_rewrite_with_its_timestamp_restored_is_recomputed(self, tmp_path):
        module = tmp_path / "m.py"
        module.write_text("safe = 1\n", encoding="utf-8")
        before = os.stat(module)
        assert derived(module, _record) == "safe"

        module.write_text("evil = 2\n", encoding="utf-8")
        os.utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = os.stat(module)
        assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)

        assert derived(module, _record) == "evil"
        assert len(_SEEN) == 2

    def test_the_tree_it_was_computed_from_is_not_kept(self, tmp_path):
        """Only the value stays: a whole-tree scan must not hold every tree."""
        module = tmp_path / "m.py"
        module.write_text("safe = 1\n", encoding="utf-8")
        assert derived(module, _record) == "safe"
        gc.collect()

        assert _SEEN[0]() is None
        assert derived(module, _record) == "safe"
        assert len(_SEEN) == 1

    def test_a_tree_source_tree_cached_is_the_one_it_reads(self, tmp_path):
        module = tmp_path / "m.py"
        module.write_text("safe = 1\n", encoding="utf-8")
        tree = source_tree(module)

        assert derived(module, _record) == "safe"
        assert _SEEN[0]() is tree

    def test_a_later_source_tree_call_keeps_the_value(self, tmp_path):
        module = tmp_path / "m.py"
        module.write_text("safe = 1\n", encoding="utf-8")
        assert derived(module, _record) == "safe"

        assert source_tree(module) is source_tree(module)
        assert derived(module, _record) == "safe"
        assert len(_SEEN) == 1
