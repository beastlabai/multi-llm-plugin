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
"""

import ast
import hashlib
import io
from pathlib import Path

#: ``skills/multi-llm``, the directory the runtime modules live under.
SKILL_ROOT = Path(__file__).resolve().parent.parent

#: path -> (sha256 of the bytes that were parsed, their tree).
_trees = {}


def source_text(path):
    """The UTF-8 text of ``path``, read from disk on every call."""
    return Path(path).read_text(encoding="utf-8")


def source_tree(path):
    """The parsed ``ast.Module`` of ``path`` (``filename`` is its path), shared: do not mutate."""
    key = str(Path(path))
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).digest()
    cached = _trees.get(key)
    if cached is None or cached[0] != digest:
        # The same decoding ``source_text`` applies (UTF-8, universal newlines).
        text = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8").read()
        cached = (digest, ast.parse(text, filename=key))
        _trees[key] = cached
    return cached[1]


def python_files(root=SKILL_ROOT):
    """Every ``*.py`` under ``root``, sorted: the unfiltered walk, rescanned on every call.

    Callers keep their own exclusions (tests, ``.venv``, vendored code) because
    each convention check defines "runtime module" slightly differently.
    """
    return sorted(Path(root).rglob("*.py"))
