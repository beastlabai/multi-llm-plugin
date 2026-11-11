"""``tests/moved_name_writes``: the guards' whole-tree walk reuses its work safely.

Every moved-names guard calls :meth:`MovedNameWriteScanner.scan_dir`, which keeps
each file's ``_candidates`` (the nodes a scan can act on) through
``source_cache.derived`` instead of parsing and walking the file again. These pin
what that reuse must not cost: a rewritten file is scanned as it now reads (even
with its size and mtime unchanged), a file added or deleted between two walks
counts, a shared tree is never changed, the candidates drop nothing a scan
reports, and nothing is skipped on a text search for the module's name. The
write forms themselves are pinned by each owner's ``TestTheGuardSees*`` class,
against its real table and message.
"""

import ast
import os

import pytest

from tests.moved_name_writes import MovedNameWriteScanner
from tests.source_cache import source_tree

#: A made-up module whose moved names are ``aaaa`` and ``cccc``; ``bbbb`` is its own.
MOVED = {"aaaa": "owner", "cccc": "owner"}
WRITE = 'from unittest import mock\nmock.patch("utils.old_mod.aaaa")\n'
CLEAN = 'from unittest import mock\nmock.patch("utils.old_mod.bbbb")\n'


def _scanner():
    return MovedNameWriteScanner(MOVED, "old_mod")


def _names(root, skip=None):
    return [(path.relative_to(root).as_posix(), hit[2])
            for path, hit in _scanner().scan_dir(root, skip=skip)]


@pytest.mark.parametrize("source, names", [
    (WRITE, ["aaaa"]),
    (CLEAN, []),
    ('from utils import old_mod as om\nom.aaaa = 1\n', ["aaaa"]),
    ('import utils.old_mod as om\nmonkeypatch.setattr(om, "aaaa", None)\n', ["aaaa"]),
    ('monkeypatch.setattr("utils.old_mod.aaaa", None)\n', ["aaaa"]),
    ('del old_mod.aaaa\n', ["aaaa"]),
    ('found = old_mod.aaaa()\n', []),
    # An alias of an alias counts only once the first is known: walk order.
    ('import utils.old_mod as om\nalias = om\nalias.aaaa = 1\n', ["aaaa"]),
    ('old_mod.cccc = 1\nold_mod.bbbb = 2\nold_mod.aaaa = 3\n', ["cccc", "aaaa"]),
    # Every value shape an alias can take, and every target shape a write can.
    ('import utils.old_mod\nalias = utils.old_mod\nalias.aaaa = 1\n', ["aaaa"]),
    ('import importlib\nalias = importlib.import_module("utils.old_mod")\nalias.aaaa = 1\n',
     ["aaaa"]),
    ('import sys\nalias = sys.modules["utils.old_mod"]\nalias.aaaa = 1\n', ["aaaa"]),
    ('alias = other = old_mod\nother.aaaa = 1\n', ["aaaa"]),
    ('alias = old_mod.thing\nalias.aaaa = 1\n', []),
    ('vars(old_mod)["aaaa"] = 1\nold_mod.__dict__["cccc"] = 2\n', ["aaaa", "cccc"]),
    ('old_mod.aaaa += 1\nold_mod.cccc: int = 2\n', ["aaaa", "cccc"]),
    ('first, *old_mod.aaaa = [1, 2]\n[old_mod.cccc, x] = 3, 4\n', ["aaaa", "cccc"]),
    ('del vars(old_mod)["aaaa"], old_mod.cccc\n', ["aaaa", "cccc"]),
    # A statement comes before the call in its value.
    ('def f():\n    old_mod.aaaa = mock.patch("utils.old_mod.cccc")\n', ["aaaa", "cccc"]),
])
def test_a_parsed_tree_gives_what_the_text_gives(source, names):
    """``scan`` (the owners' self-tests) and ``scan_tree`` (the guards) are one scan,
    reporting in walk order. Both read only the nodes ``_candidates`` keeps, so
    these also pin that it keeps every shape a scan acts on."""
    from_tree = _scanner().scan_tree(ast.parse(source))
    assert from_tree == _scanner().scan(source)
    assert [hit[2] for hit in from_tree] == names


@pytest.mark.parametrize("spelling", [
    '"utils.old" "_mod.aaaa"',
    '"utils.old\\x5fmod.aaaa"',
])
def test_a_target_spelled_without_the_module_name_is_still_seen(tmp_path, spelling):
    """``ast.parse`` joins and decodes these; a text search for ``old_mod`` would
    skip the file. The walk must not take that shortcut."""
    source = f"from unittest import mock\nmock.patch({spelling})\n"
    assert "old_mod" not in source
    (tmp_path / "test_spelled.py").write_text(source, encoding="utf-8")
    assert _names(tmp_path) == [("test_spelled.py", "aaaa")]


def test_a_rewrite_that_keeps_size_and_mtime_is_scanned_as_it_now_reads(tmp_path):
    """A clean file's cached tree must not hide a write added to it later."""
    module = tmp_path / "test_x.py"
    module.write_text(CLEAN, encoding="utf-8")
    before = os.stat(module)
    assert _names(tmp_path) == []

    module.write_text(WRITE, encoding="utf-8")
    os.utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(module)
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)

    assert _names(tmp_path) == [("test_x.py", "aaaa")]


def test_a_file_added_or_deleted_between_two_walks_counts(tmp_path):
    (tmp_path / "test_clean.py").write_text(CLEAN, encoding="utf-8")
    assert _names(tmp_path) == []

    (tmp_path / "support").mkdir()
    (tmp_path / "support" / "helper.py").write_text(WRITE, encoding="utf-8")
    assert _names(tmp_path) == [("support/helper.py", "aaaa")]

    (tmp_path / "support" / "helper.py").unlink()
    assert _names(tmp_path) == []


def test_only_the_skipped_file_is_left_out(tmp_path):
    for name in ("test_guard.py", "test_other.py"):
        (tmp_path / name).write_text(WRITE, encoding="utf-8")
    assert _names(tmp_path, skip=tmp_path / "test_guard.py") == [("test_other.py", "aaaa")]
    assert _names(tmp_path) == [("test_guard.py", "aaaa"), ("test_other.py", "aaaa")]


def _snapshot(tree):
    """Fields, positions and any attribute stamped on a node (``ast.dump`` shows no
    stamped attribute)."""
    return (ast.dump(tree, include_attributes=True),
            [(type(node).__name__, sorted(vars(node))) for node in ast.walk(tree)])


def test_a_walk_leaves_the_shared_tree_as_it_found_it(tmp_path):
    """A tree another check already cached is the one the walk reads, and the one
    every other caller in the process gets back."""
    module = tmp_path / "test_x.py"
    module.write_text(WRITE + "a, old_mod.aaaa = 1, 2\ndel old_mod.aaaa\n", encoding="utf-8")
    tree = source_tree(module)
    before = _snapshot(tree)

    assert [name for _, name in _names(tmp_path)] == ["aaaa", "aaaa", "aaaa"]

    assert source_tree(module) is tree
    assert _snapshot(tree) == before


def test_a_second_walk_parses_nothing_again(tmp_path, monkeypatch):
    """The point of the cache: several guards in one process pay for one parse."""
    for name in ("test_a.py", "test_b.py"):
        (tmp_path / name).write_text(WRITE, encoding="utf-8")
    parsed = []
    real_parse = ast.parse

    def counting_parse(source, filename="<unknown>", *args, **kwargs):
        parsed.append(str(filename))
        return real_parse(source, filename, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", counting_parse)
    first = _names(tmp_path)
    second = MovedNameWriteScanner({"bbbb": "owner"}, "old_mod")
    assert [path.name for path, _ in second.scan_dir(tmp_path, skip=None)] == []
    assert _names(tmp_path) == first == [("test_a.py", "aaaa"), ("test_b.py", "aaaa")]
    assert sorted(parsed) == sorted(str(tmp_path / n) for n in ("test_a.py", "test_b.py"))
