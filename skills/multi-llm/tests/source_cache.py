"""One parsed-source cache for the tests that read this repo's own source.

Architecture and convention checks (``ast`` walks over runtime modules, "no module
does X" scans) used to each read and parse the same files. They go through here
instead: a file is parsed once per version of its content per pytest process,
however many tests ask for it. Not a test module itself.

* Freshness never rests on timestamps. Every call reads the file again (reading is
  cheap, parsing is not) and a tree is reused only while the bytes hash the same,
  so a test that rewrites a file between two calls gets the new text and tree,
  even when the rewrite keeps the length and restores the mtime.
* Directory membership is rescanned on every call (a walk of this tree takes
  milliseconds), so a module added or deleted mid-process shows in the next walk.
* A tree is shared between callers: walk it, never mutate it (no
  ``NodeTransformer``, no ``ast.fix_missing_locations``, no attribute stamping).
  A test that needs to modify a tree parses its own copy with ``ast.parse``.
* The latest tree of each path stays cached for the life of the process, which is
  why the import-layer check, which wants memory kept down, parses one module at a
  time and does not use this.
* ``derived`` keeps a value computed from a tree (the nodes a scan reads) instead
  of the tree, for a scan over so much source that keeping every tree would cost
  more than parsing it again: each live node makes every garbage collection
  slower, and pytest and the interpreter run several on the way out. The value is
  kept per version of the bytes, like a tree.
"""

import ast
import hashlib
import io
from pathlib import Path

#: ``skills/multi-llm``, the directory the runtime modules live under.
SKILL_ROOT = Path(__file__).resolve().parent.parent

#: path -> [sha256 of the file's bytes, their tree or None until ``source_tree``
#: asks for it, {function: its ``derived`` value}].
_trees = {}


def source_text(path):
    """The UTF-8 text of ``path``, read from disk on every call."""
    return Path(path).read_text(encoding="utf-8")


def _entry(path):
    """``(key, the file's bytes, its entry)``: a fresh entry when the bytes changed."""
    key = str(Path(path))
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).digest()
    entry = _trees.get(key)
    if entry is None or entry[0] != digest:
        entry = _trees[key] = [digest, None, {}]
    return key, data, entry


def _parse(key, data):
    # The same decoding ``source_text`` applies (UTF-8, universal newlines).
    text = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8").read()
    return ast.parse(text, filename=key)


def source_tree(path):
    """The parsed ``ast.Module`` of ``path`` (``filename`` is its path), shared: do not mutate."""
    key, data, entry = _entry(path)
    if entry[1] is None:
        entry[1] = _parse(key, data)
    return entry[1]


def derived(path, function):
    """``function(tree)`` for ``path``'s tree, computed once per version of its bytes.

    *function* is a module-level function (it is the cache key), only reads the
    tree, and depends on nothing else, since its value is reused for as long as
    the bytes hash the same. It gets the cached tree if ``source_tree`` already
    parsed one; otherwise a tree parsed for it alone, which is not kept. The value
    is shared like a tree: do not mutate it.
    """
    key, data, entry = _entry(path)
    values = entry[2]
    if function not in values:
        tree = entry[1] if entry[1] is not None else _parse(key, data)
        values[function] = function(tree)
    return values[function]


def python_files(root=SKILL_ROOT):
    """Every ``*.py`` under ``root``, sorted: the unfiltered walk, rescanned on every call.

    Callers keep their own exclusions (tests, ``.venv``, vendored code) because
    each convention check defines "runtime module" slightly differently.
    """
    return sorted(Path(root).rglob("*.py"))
