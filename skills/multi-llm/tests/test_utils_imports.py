"""Every ``utils`` module imports as part of the package, alone, in a fresh interpreter.

``utils/__init__.py`` re-exports nothing (docs/development/dispatcher.md#import-safety):
a module imports each name from the module that defines it, so importing a leaf
loads only what that leaf needs. Each module here is the first import of its own
interpreter, which is where a load-time cycle or an import that only worked
because something else loaded first shows up.
"""

import ast
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

SKILL_ROOT = Path(__file__).resolve().parent.parent
UTILS = SKILL_ROOT / "utils"

#: Leaf helpers that must not load the provider stack or a YAML parser: the
#: plain text helper, and the Stop-hook check a host runs at every turn end.
LIGHT_LEAVES = ("utils.text_clip", "utils.report_check")

#: The executor's facade and the modules below it
#: (docs/development/delegation.md#where-the-executors-helpers-live).
EXECUTOR_MODULES = (
    "subagent_executor", "batch_reply", "failure_capture", "task_outcome", "task_result",
    "runner_selection", "probe_state", "sandbox_walk", "dispatch_spawn",
    "attempt_classifier", "runner_preflight", "task_dispatch", "task_batch",
    "task_file_runner", "known_checks",
)

_PROBE = (
    "import importlib, json, sys\n"
    "importlib.import_module(sys.argv[1])\n"
    "print(json.dumps(sorted(sys.modules)))\n"
)


def _import_targets(node: ast.AST, package: str) -> List[str]:
    """The modules an import statement names, relative spellings resolved."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    assert isinstance(node, ast.ImportFrom)
    base = package.split(".")
    base = base[: len(base) - (node.level - 1)] if node.level else []
    module = ".".join(base + ([node.module] if node.module else []))
    targets = []
    for alias in node.names:
        sub = f"{module}.{alias.name}"
        is_module = (SKILL_ROOT / Path(*sub.split("."))).with_suffix(".py").is_file()
        targets.append(sub if is_module else module)
    return targets


def _top_and_function_imports(path: Path, package: str):
    """``(modules imported at module level, [(line, module) imported in a function]``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top, nested = set(), []

    def walk(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                if isinstance(child, ast.ImportFrom) and child.module == "__future__":
                    continue
                for target in _import_targets(child, package):
                    if in_function:
                        nested.append((child.lineno, target))
                    else:
                        top.add(target)
            walk(child, in_function or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))

    walk(tree, False)
    return top, nested


def _module_names() -> List[str]:
    names = []
    for path in sorted(UTILS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = path.relative_to(SKILL_ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def _fresh_import(name: str) -> Tuple[str, int, str, List[str]]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, name],
        cwd=SKILL_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    loaded: List[str] = []
    if done.returncode == 0:
        loaded = json.loads(done.stdout.strip().splitlines()[-1])
    return name, done.returncode, done.stderr, loaded


@lru_cache(maxsize=1)
def _results() -> Dict[str, Tuple[int, str, List[str]]]:
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        return {
            name: (code, stderr, loaded)
            for name, code, stderr, loaded in pool.map(_fresh_import, _module_names())
        }


def test_the_package_re_exports_nothing():
    tree = ast.parse((UTILS / "__init__.py").read_text(encoding="utf-8"))
    statements = [
        node for node in tree.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str))
    ]
    assert statements == [], (
        "utils/__init__.py holds only its docstring: import a name from the "
        "module that defines it (`from utils.git_utils import get_staged_files`)"
    )


def test_the_module_list_is_not_vacuous():
    names = _module_names()
    assert "utils.text_clip" in names and "utils.providers.base" in names
    assert len(names) > 100


def test_every_module_imports_first_in_a_fresh_interpreter():
    failed = {
        name: (stderr.strip().splitlines() or ["(no stderr)"])[-1]
        for name, (code, stderr, _loaded) in _results().items()
        if code != 0
    }
    assert failed == {}, "\n".join(f"{name}: {line}" for name, line in sorted(failed.items()))


def test_a_light_leaf_loads_no_provider_and_no_yaml():
    for name in LIGHT_LEAVES:
        code, stderr, loaded = _results()[name]
        assert code == 0, stderr
        heavy = [m for m in loaded if m == "yaml" or m.startswith("utils.providers")]
        assert heavy == [], f"importing {name} loads {heavy}"


def test_the_executor_imports_no_module_twice():
    """A module imported at the top is not imported again inside a function:
    a second, lazy import of it guards no cycle and hides the real dependency
    (docs/development/dispatcher.md#import-safety)."""
    repeats = []
    for name in EXECUTOR_MODULES:
        top, nested = _top_and_function_imports(UTILS / f"{name}.py", "utils")
        repeats += [f"utils/{name}.py:{line} {target}" for line, target in nested if target in top]
    assert repeats == [], "\n".join(repeats)
