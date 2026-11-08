#!/usr/bin/env python3
"""Which tests cover a source module: a static import map, advisory only.

The suite is hundreds of ``tests/test_*.py`` files whose names do not follow the
module layout, so "which tests cover ``utils/foo.py``" is not answerable from a
listing. This script answers it by parsing every test file with ``ast`` and
recording which source modules each one imports directly (``import x``,
``from x import y``, ``from utils import foo`` where ``foo`` is a module, and a
literal dynamic import: ``importlib.import_module("utils.x")`` or
``__import__("utils.x")``, under any ``importlib`` alias). Stdlib only, no
spend, nothing is imported or executed.

Two small declared tables add what an import scan cannot see (each row has a
reason, and ``tests/test_module_tests_map.py`` keeps them honest):

* ``SWEEPS``: a test that loads modules by a *computed* name, e.g. every
  ``utils.providers.*`` adapter in a loop;
* ``HARNESS_SUITES``: every ``tests/test_e2e_*.py``. These drive the skill as a
  subprocess, so they import nothing; the entry points a suite runs are derived
  from the mode names and orchestrator names it passes, and the rest is declared.

It is **advisory**. The map still misses coverage through ``runpy``, through a
helper module that re-exports a source module, and through another source
module that imports the one you changed; it counts a hub like
``utils.state_manager`` as covered by most of the suite. Use it to pick a
focused first run; the CI command stays the gate.

Usage (from the repository root)::

    python skills/multi-llm/tests/tools/tests_for.py utils/status_file.py
    python skills/multi-llm/tests/tools/tests_for.py utils.status_file --top 5
    python skills/multi-llm/tests/tools/tests_for.py utils/status_file.py --paths
    python skills/multi-llm/tests/tools/tests_for.py --write   # regenerate tests/MODULE_TESTS.md
    python skills/multi-llm/tests/tools/tests_for.py --check   # exit 1 if that map is stale

``make test-for MODULE=utils/status_file.py`` prints the list and runs only
those files; ``make tests-map`` runs ``--write``. See ``tests/README.md``.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

TESTS_DIR = Path(__file__).resolve().parent.parent
SKILL_ROOT = TESTS_DIR.parent
#: Path of the committed map, relative to the skill root (what messages print).
MAP_RELATIVE = "tests/MODULE_TESTS.md"
MAP_PATH = SKILL_ROOT / MAP_RELATIVE
#: The command a reader runs to regenerate the map, from the repository root.
WRITE_COMMAND = "make tests-map"
#: A module imported by more than this many test files cannot be narrowed.
HUB_THRESHOLD = 40
#: Test files listed per module in the committed map.
MAP_TOP = 12


class Importer(NamedTuple):
    """One test file's link to one module."""

    test: str  # file name under tests/, e.g. ``test_status_file.py``
    names: bool  # True when it does ``from <module> import <names>``
    #: ``import`` (a static import), ``dynamic`` (a literal ``import_module`` /
    #: ``__import__``), ``sweep`` (a declared computed load), ``harness`` (a
    #: subprocess suite that drives the module's entry point).
    via: str = "import"


class Sweep(NamedTuple):
    """A test file that loads modules by a computed name."""

    test: str
    patterns: Tuple[str, ...]  # fnmatch patterns over dotted module names
    reason: str


class HarnessSuite(NamedTuple):
    """A ``test_e2e_*`` suite that drives the skill as a subprocess."""

    modules: Tuple[str, ...]  # modules its runs exercise beyond the derived entry points
    reason: str


#: Glob of the suites that run the skill as a subprocess.
HARNESS_GLOB = "test_e2e_*.py"

#: Tests that load modules through a computed name (a loop over a roster), which
#: no literal-import scan can see. One row per file, with the reason.
SWEEPS: Tuple[Sweep, ...] = (
    Sweep(
        "test_retry_ladder.py", ("utils.providers.*",),
        "parametrized over adapter modules, loaded by dotted name from the parameters",
    ),
    Sweep(
        "test_reasoning_effort_ladder.py", ("utils.providers.*",),
        "loads the adapter module behind every registry id through the roster",
    ),
    Sweep(
        "test_pi_family_stream.py",
        ("utils.providers.base", "utils.providers.pi_family_stream",
         "utils.providers.pi", "utils.providers.omp"),
        "imports the pi-family modules in several orders in a fresh interpreter, "
        "by a name built from the order argument",
    ),
)

#: One row per ``tests/test_e2e_*.py``: what the suite exercises beyond the entry
#: points derived from the mode and orchestrator names it passes. A new e2e suite
#: without a row fails ``tests/test_module_tests_map.py``.
HARNESS_SUITES: Dict[str, HarnessSuite] = {
    "test_e2e_apply_code_fixes.py": HarnessSuite((), "apply_code_fixes_orchestrator subprocess"),
    "test_e2e_batching.py": HarnessSuite((), "imports the batchers directly; the scan sees them"),
    "test_e2e_code_review.py": HarnessSuite((), "code_review_orchestrator subprocess and multi_llm.py"),
    "test_e2e_commit_modes.py": HarnessSuite(
        ("utils.commit_review", "utils.commit_apply", "utils.commit_brief", "utils.commit_range"),
        "multi_llm.py subprocesses for the three commit modes: the front door, the brief, "
        "the range and the apply path run for real",
    ),
    "test_e2e_exec_subagents.py": HarnessSuite(
        ("utils.subagent_executor",), "--exec-subagents drives the executor in the subprocess",
    ),
    "test_e2e_failures.py": HarnessSuite((), "review_plan and code_review subprocesses on failing providers"),
    "test_e2e_full_chain.py": HarnessSuite((), "the phase chain through the orchestrator subprocesses"),
    "test_e2e_html_report.py": HarnessSuite(
        ("utils.html_report_generator",), "the HTML report written and reimported by a phase run",
    ),
    "test_e2e_improve_modes.py": HarnessSuite(
        ("utils.improve_workflow", "utils.improve_steps", "utils.commit_review"),
        "improve-plan, improve-code and improve-commits through multi_llm.py",
    ),
    "test_e2e_mock_infrastructure.py": HarnessSuite((), "tests the mock provider binaries, not a source module"),
    "test_e2e_runs_retention.py": HarnessSuite((), "tests the container harness's run pruning, not a source module"),
    "test_e2e_validation_edge_cases.py": HarnessSuite((), "imports utils.validation directly; the scan sees it"),
    "test_e2e_validation.py": HarnessSuite((), "implement_orchestrator subprocess on validation edge cases"),
    "test_e2e_workflow.py": HarnessSuite((), "review_plan and implement subprocesses end to end"),
}


def source_modules(root: Path = SKILL_ROOT) -> Dict[str, str]:
    """Dotted name -> skill-root-relative path, for every importable source file.

    Root scripts, ``utils/`` (and ``utils/providers/``) and ``hosts/``. A package
    ``__init__.py`` is keyed by the package name. ``tests/`` and the generated or
    vendored trees are not source modules.
    """
    modules: Dict[str, str] = {}
    for path in sorted(root.glob("*.py")):
        modules[path.stem] = path.relative_to(root).as_posix()
    for package in ("utils", "hosts"):
        for path in sorted((root / package).rglob("*.py")):
            relative = path.relative_to(root)
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            modules[".".join(parts)] = relative.as_posix()
    return modules


def _importlib_names(tree: ast.AST) -> Tuple[Set[str], Set[str]]:
    """Names bound to ``importlib`` and to its loader functions in one file.

    Returns ``(modules, loaders)``: ``import importlib as il`` binds ``il`` to the
    module, ``from importlib import import_module as load`` binds ``load`` to a
    loader. ``__import__`` is always a loader.
    """
    modules: Set[str] = set()
    loaders: Set[str] = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    modules.add(alias.asname or "importlib")
                elif alias.name.startswith("importlib.") and not alias.asname:
                    modules.add("importlib")  # ``import importlib.util`` binds ``importlib``
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    loaders.add(alias.asname or alias.name)
    return modules, loaders


def _literal(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_names(node: Optional[ast.AST]) -> List[str]:
    if isinstance(node, (ast.List, ast.Tuple)):
        return [value for value in map(_literal, node.elts) if value]
    return []


def _dynamic_imports(tree: ast.AST) -> List[str]:
    """Dotted names a file loads through a literal ``import_module`` / ``__import__``.

    ``importlib.import_module("utils.x")`` under any alias of ``importlib``, a
    ``from importlib import import_module`` name, and ``__import__("utils.x")``,
    plus each literal ``fromlist`` entry of ``__import__`` as ``utils.x.<entry>``.
    A computed or relative name is not a literal and is skipped; ``SWEEPS`` is the
    place to declare a computed one.
    """
    modules, loaders = _importlib_names(tree)
    found: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        dunder = isinstance(func, ast.Name) and func.id == "__import__"
        if isinstance(func, ast.Name):
            loader = func.id in loaders
        else:
            loader = (
                isinstance(func, ast.Attribute) and func.attr == "import_module"
                and isinstance(func.value, ast.Name) and func.value.id in modules
            )
        if not loader:
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        name = _literal(node.args[0] if node.args else keywords.get("name"))
        if not name or name.startswith("."):
            continue
        found.append(name)
        if dunder:
            fromlist = node.args[3] if len(node.args) > 3 else keywords.get("fromlist")
            found.extend(f"{name}.{entry}" for entry in _literal_names(fromlist))
    return found


def _imports_of(tree: ast.AST, modules: Dict[str, str]) -> Dict[str, Tuple[bool, str]]:
    """Source modules a parsed test file imports -> (via ``from ... import names``, how).

    ``how`` is ``import`` for a static import and ``dynamic`` for a literal
    ``import_module`` / ``__import__``; a static import wins when a file has both.
    """
    found: Dict[str, Tuple[bool, str]] = {}

    def note(module: str, names: bool, via: str = "import") -> None:
        if module not in modules:
            return
        old_names, old_via = found.get(module, (False, "dynamic"))
        # A from-import of names anywhere in the file wins over a bare import,
        # and a static import over a dynamic one.
        found[module] = (old_names or names, "import" if "import" in (old_via, via) else via)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                note(alias.name, False)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            base = node.module
            note(base, True)
            for alias in node.names:
                # ``from utils import foo`` imports the module ``utils.foo``.
                note(f"{base}.{alias.name}", False)
    for name in _dynamic_imports(tree):
        note(name, False, "dynamic")
    return found


def _mode_entry_points(root: Path) -> Dict[str, str]:
    """Mode word -> orchestrator module, read from ``modes.py`` without importing it."""
    try:
        tree = ast.parse((root / "modes.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    entries: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Mode":
            keywords = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
            name, target = keywords.get("name"), keywords.get("target_module")
            if name and target and target.endswith("_orchestrator"):
                entries[name] = target
    return entries


def _runner_entry_points(tests_dir: Path) -> Dict[str, str]:
    """``SkillRunner.ORCHESTRATORS`` key -> orchestrator module (the harness roster)."""
    try:
        tree = ast.parse((tests_dir / "harness" / "skill_runner.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
            and any(getattr(t, "id", "") == "ORCHESTRATORS" for t in node.targets)
        ):
            pairs = zip(map(_literal, node.value.keys), map(_literal, node.value.values))
            return {k: v[:-3] if v.endswith(".py") else v for k, v in pairs if k and v}
    return {}


def _harness_entry_points(
    tree: ast.AST, modes: Dict[str, str], runner: Dict[str, str], modules: Dict[str, str]
) -> Set[str]:
    """Orchestrator modules a harness suite drives, from the strings it passes.

    A mode word (``"review-commits"``, also as ``--review-commits``), a
    ``SkillRunner`` orchestrator key (``"code_review"``) or an orchestrator file
    name (``"code_review_orchestrator.py"``).
    """
    found: Set[str] = set()
    for node in ast.walk(tree):
        text = _literal(node)
        if not text:
            continue
        word = text[2:] if text.startswith("--") else text
        stem = word[:-3] if word.endswith(".py") else word
        for target in (modes.get(word), runner.get(text)):
            if target in modules:
                found.add(target)
        if stem.endswith("_orchestrator") and stem in modules:
            found.add(stem)
    return found


def build_map(
    tests_dir: Path = TESTS_DIR, modules: Optional[Dict[str, str]] = None,
    sweeps: Sequence[Sweep] = SWEEPS,
    suites: Optional[Dict[str, HarnessSuite]] = None,
) -> Dict[str, List[Importer]]:
    """Source module (dotted) -> the test files that cover it, sorted by file name.

    Edges: static imports, literal dynamic imports, the declared ``sweeps``, and
    for each ``test_e2e_*`` suite the orchestrators it drives (derived) plus the
    modules its ``HARNESS_SUITES`` row declares. A sweep or suite row naming a test
    file absent from ``tests_dir`` is skipped, so a scratch tree is unaffected.
    """
    modules = modules if modules is not None else source_modules(tests_dir.parent)
    suites = HARNESS_SUITES if suites is None else suites
    modes = _mode_entry_points(tests_dir.parent)
    runner = _runner_entry_points(tests_dir)
    edges: Dict[str, Dict[str, Importer]] = {name: {} for name in modules}

    def add(module: str, test: str, names: bool, via: str) -> None:
        old = edges[module].get(test)
        if old is None or (old.via != "import" and via == "import"):
            edges[module][test] = Importer(test, names or bool(old and old.names), via)

    trees: Dict[str, ast.AST] = {}
    for path in sorted(tests_dir.glob("test_*.py")):
        try:
            trees[path.name] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue  # pytest reports a broken test file; the map just skips it
        for module, (names, via) in _imports_of(trees[path.name], modules).items():
            add(module, path.name, names, via)
    for sweep in sweeps:
        if sweep.test in trees:
            for module in modules:
                if any(fnmatch.fnmatchcase(module, pattern) for pattern in sweep.patterns):
                    add(module, sweep.test, False, "sweep")
    for name, tree in trees.items():
        if not fnmatch.fnmatchcase(name, HARNESS_GLOB):
            continue
        driven = _harness_entry_points(tree, modes, runner, modules)
        if name in suites:
            driven |= {m for m in suites[name].modules if m in modules}
        for module in driven:
            add(module, name, False, "harness")
    return {name: sorted(found.values(), key=lambda item: item.test) for name, found in edges.items()}


def _name_tier(test: str, module: str) -> int:
    """0 ``test_<leaf>.py``, 1 ``test_<leaf>_*``, 2 leaf in the name, 3 no match."""
    leaf = module.rsplit(".", 1)[-1]
    stem = test[:-3] if test.endswith(".py") else test
    if stem == f"test_{leaf}":
        return 0
    if stem.startswith(f"test_{leaf}_"):
        return 1
    if leaf in stem:
        return 2
    return 3


def ranked(
    module: str, importers: Sequence[Importer], tests_dir: Path = TESTS_DIR,
    by_size: bool = True,
) -> List[Importer]:
    """Name match first, then a declared route, a from-import of names, then smaller files.

    A declared route (a sweep or a harness suite) is a deliberate association, so it
    ranks above a plain import: on a hub such as an orchestrator it keeps the
    subprocess suite inside the committed map's top files.

    ``by_size=False`` drops the size key (ties fall to the file name). The
    committed map uses it: a size key would make the map stale on every test
    edit, not only when the import graph changed.
    """
    def key(item: Importer) -> Tuple[int, int, int, int, str]:
        size = 0
        if by_size:
            try:
                size = (tests_dir / item.test).stat().st_size
            except OSError:
                size = 0
        routed = 0 if item.via in ("sweep", "harness") else 1
        return (_name_tier(item.test, module), routed, 0 if item.names else 1, size, item.test)

    return sorted(importers, key=key)


def resolve(query: str, modules: Dict[str, str]) -> Optional[str]:
    """Accept ``utils/foo.py``, ``utils.foo``, ``skills/multi-llm/utils/foo.py``, ``foo.py``."""
    text = query.strip().replace("\\", "/")
    if text.startswith("skills/multi-llm/"):
        text = text[len("skills/multi-llm/"):]
    if text.endswith(".py"):
        text = text[:-3]
    text = text.strip("/").replace("/", ".")
    if text.endswith(".__init__"):
        text = text[: -len(".__init__")]
    return text if text in modules else None


def _is_hub(count: int) -> bool:
    return count > HUB_THRESHOLD


def _hub_note(count: int) -> str:
    return (
        f"hub: {count} test files cover this module (more than {HUB_THRESHOLD}); "
        "the import map cannot narrow it, so pick by name or run the CI command"
    )


def render_query(
    module: str, modules: Dict[str, str], mapping: Dict[str, List[Importer]],
    top: Optional[int] = None,
) -> str:
    importers = ranked(module, mapping[module])
    shown = importers if top is None else importers[:top]
    lines = [f"{modules[module]}: {len(importers)} test file(s) cover it by import, dynamic import or a declared route"]
    if _is_hub(len(importers)):
        lines.append(f"  {_hub_note(len(importers))}")
    if not importers:
        lines.append(
            "  none; it may be covered only indirectly (through another module "
            "or a CLI run), so grep tests/ for its name"
        )
    for item in shown:
        tier = _name_tier(item.test, module)
        tags = []
        if tier < 3:
            tags.append("name match")
        if item.names:
            tags.append("from-import")
        if item.via != "import":
            tags.append({"dynamic": "dynamic import", "sweep": "declared sweep",
                         "harness": "e2e harness"}.get(item.via, item.via))
        suffix = f"  ({', '.join(tags)})" if tags else ""
        lines.append(f"  tests/{item.test}{suffix}")
    if top is not None and len(importers) > top:
        lines.append(f"  ... +{len(importers) - top} more (drop --top to list them)")
    lines.append(
        "  advisory: an import map misses indirect coverage; it never replaces "
        "the CI command."
    )
    return "\n".join(lines) + "\n"


def render_map(
    modules: Dict[str, str], mapping: Dict[str, List[Importer]],
    tests_dir: Path = TESTS_DIR,
    sweeps: Sequence[Sweep] = SWEEPS,
    suites: Optional[Dict[str, HarnessSuite]] = None,
) -> str:
    """The committed ``MODULE_TESTS.md``: deterministic, no sizes, no timestamps.

    Nothing here depends on how many test files exist: only the mapping itself, so
    an unrelated new test leaves the file unchanged.
    """
    suites = HARNESS_SUITES if suites is None else suites
    lines = [
        "<!-- GENERATED FILE - DO NOT EDIT.",
        "     Regenerate with `make tests-map` (from the repository root), or",
        "     `python skills/multi-llm/tests/tools/tests_for.py --write`.",
        "     tests/test_module_tests_map.py fails when this file is stale, i.e. when a",
        "     test file's imports or a declared route change the mapping. -->",
        "",
        "# Module to tests map",
        "",
        "Which `tests/test_*.py` files cover each source module, from a static AST scan:",
        "`import x`, `from x import y`, `from utils import foo`, a literal",
        "`importlib.import_module(\"x\")` / `__import__(\"x\")`, plus two declared tables in",
        "`tests/tools/tests_for.py` (listed at the end). **Advisory:** the scan still",
        "misses coverage through `runpy`, through a helper module that re-exports a",
        "source module, and through another source module that imports the one you",
        "changed; it never replaces the CI command. For a ranked list that also weighs",
        "file size, and to run just those files, use `make test-for MODULE=utils/foo.py`.",
        "",
        "Files are ordered by name match, then a `from x import names` import, then name;",
        f"at most {MAP_TOP} are listed per module. A **hub** (more than {HUB_THRESHOLD} files)",
        "cannot be narrowed by this map: choose by the file names or run the CI command.",
        "",
        "| Module | Test files | Top files |",
        "|--------|-----------:|-----------|",
    ]
    for module in sorted(modules, key=lambda name: modules[name]):
        importers = ranked(module, mapping[module], tests_dir, by_size=False)
        count = len(importers)
        names = [f"`{item.test}`" for item in importers[:MAP_TOP]]
        if count > MAP_TOP:
            names.append(f"+{count - MAP_TOP} more")
        if _is_hub(count):
            cell = "**hub**, cannot narrow: " + ", ".join(names)
        elif not names:
            cell = "none (no direct importer)"
        else:
            cell = ", ".join(names)
        lines.append(f"| `{modules[module]}` | {count} | {cell} |")
    lines += [
        "",
        "## Declared routes",
        "",
        "Tests that load modules by a computed name (`SWEEPS`):",
        "",
        "| Test file | Covers | Why |",
        "|-----------|--------|-----|",
    ]
    for sweep in sweeps:
        covers = ", ".join(f"`{pattern}`" for pattern in sweep.patterns)
        lines.append(f"| `{sweep.test}` | {covers} | {sweep.reason} |")
    lines += [
        "",
        "Subprocess suites (`HARNESS_SUITES`). The orchestrators a suite drives are",
        "derived from the mode words and orchestrator names it passes; the modules",
        "column is what the row adds:",
        "",
        "| Test file | Adds | Why |",
        "|-----------|------|-----|",
    ]
    for name in sorted(suites):
        added = ", ".join(f"`{module}`" for module in suites[name].modules) or "-"
        lines.append(f"| `{name}` | {added} | {suites[name].reason} |")
    lines.append("")
    return "\n".join(lines)


def generate(tests_dir: Path = TESTS_DIR) -> str:
    modules = source_modules(tests_dir.parent)
    return render_map(modules, build_map(tests_dir, modules), tests_dir)


def check(map_path: Path = MAP_PATH, tests_dir: Path = TESTS_DIR) -> Optional[str]:
    """None when the committed map is fresh, else the one-line message to print."""
    try:
        current = map_path.read_text(encoding="utf-8")
    except OSError:
        current = None
    if current == generate(tests_dir):
        return None
    return (
        f"ERROR: {MAP_RELATIVE} is stale (the module-to-tests mapping changed). "
        f"Run `{WRITE_COMMAND}` and commit the result."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("module", nargs="?", help="module path or dotted name, e.g. utils/foo.py")
    parser.add_argument("--write", action="store_true", help=f"regenerate {MAP_RELATIVE}")
    parser.add_argument("--check", action="store_true", help=f"exit 1 if {MAP_RELATIVE} is stale")
    parser.add_argument("--top", type=int, metavar="N", help="list only the first N files")
    parser.add_argument(
        "--paths", action="store_true",
        help="print only repo-root-relative test paths, one per line (for make)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.write:
        MAP_PATH.write_bytes(generate().encode("utf-8"))
        print(f"wrote {MAP_RELATIVE}")
        return 0
    if args.check:
        problem = check()
        if problem:
            print(problem, file=sys.stderr)
            return 1
        print(f"{MAP_RELATIVE} is up to date.")
        return 0
    if not args.module:
        parser.error("give a module (utils/foo.py), or --write / --check")

    modules = source_modules()
    module = resolve(args.module, modules)
    if module is None:
        close = difflib.get_close_matches(
            args.module.replace("\\", "/").removesuffix(".py"),
            [path[:-3] for path in modules.values()], n=3,
        )
        hint = f" Did you mean: {', '.join(c + '.py' for c in close)}?" if close else ""
        print(f"ERROR: {args.module!r} is not a source module under skills/multi-llm/.{hint}",
              file=sys.stderr)
        return 2
    mapping = build_map(TESTS_DIR, modules)
    if args.paths:
        items = ranked(module, mapping[module])
        if args.top is not None:
            items = items[: args.top]
        for item in items:
            print(f"skills/multi-llm/tests/{item.test}")
        return 0
    sys.stdout.write(render_query(module, modules, mapping, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
