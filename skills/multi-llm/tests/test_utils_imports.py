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
from typing import Dict, List, Optional, Set, Tuple

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
UTILS = SKILL_ROOT / "utils"

#: Modules that must not load the provider stack or a YAML parser when imported:
#: the plain text helper, the hook commands a host runs at every turn end or tool
#: call, ``doctor`` (its 1.0 s budget), the sub-second ``status`` and
#: ``next-command`` paths with the run records they read, and the light ``util``
#: commands. Their function-level imports of heavier modules are what keeps them
#: so (docs/development/dispatcher.md#import-safety).
LIGHT_LEAVES = (
    "utils.text_clip", "utils.report_check", "utils.wakeup_check", "utils.check_reply",
    "utils.doctor", "utils.metrics", "utils.next_phase", "utils.run_status",
    "utils.status_commands", "utils.status_native", "utils.status_text", "utils.status_views",
    "utils.resume_state", "utils.run_sweep", "utils.detach", "utils.progress_tracker",
    "utils.host_install", "utils.html_report_generator", "utils.prior_changes", "utils.backup",
    "utils.batch_reply", "utils.check_records",
)

#: The same rule for the dispatcher and the entry scripts that front other
#: modes: each is imported first in a fresh interpreter too.
LIGHT_SCRIPTS = (
    "multi_llm", "full_workflow_orchestrator", "check_workflow_prerequisites",
    "review_commits_orchestrator", "apply_commit_fixes_orchestrator",
    "improve_commits_orchestrator",
)

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


def _type_checking_names(tree: ast.Module) -> Set[str]:
    """``TYPE_CHECKING`` and each name the module binds it to
    (``from typing import TYPE_CHECKING as TC``)."""
    names = {"TYPE_CHECKING"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("typing", "typing_extensions"):
            names.update(alias.asname or alias.name for alias in node.names
                         if alias.name == "TYPE_CHECKING")
    return names


def _only_while_type_checking(test: ast.expr, names: Set[str]) -> Optional[bool]:
    """True for a test that holds only under a type checker (``TYPE_CHECKING``,
    an alias of it, ``typing.TYPE_CHECKING``), False for its negation, None for
    a test decided at run time."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _only_while_type_checking(test.operand, names)
        return None if inner is None else not inner
    if (isinstance(test, ast.Name) and test.id in names) or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"):
        return True
    return None


_TRY_NODES = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())


def _top_and_function_imports(path: Path, package: str):
    """``(modules the module imports whenever it loads, [(line, module) imported in a function]``.

    Only an import that runs every time the module loads counts as a top-level
    one. An import under a run-time ``if`` (a platform test), a loop, a ``with``,
    or a ``try`` that handles exceptions (``except ImportError``) may not run,
    so a function importing the same module again repeats nothing. The body
    of ``if TYPE_CHECKING:`` (under any alias) never runs and counts nowhere;
    its ``else`` runs every time.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    checking = _type_checking_names(tree)
    top, nested = set(), []

    def maybe(scope: str) -> str:
        return scope if scope == "function" else "conditional"

    def visit(statements: List[ast.stmt], scope: str) -> None:
        for node in statements:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    continue
                for target in _import_targets(node, package):
                    if scope == "function":
                        nested.append((node.lineno, target))
                    elif scope == "top":
                        top.add(target)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(node.body, "function")
            elif isinstance(node, ast.ClassDef):
                visit(node.body, scope)
            elif isinstance(node, ast.If):
                checker_only = _only_while_type_checking(node.test, checking)
                if checker_only is None:
                    visit(node.body, maybe(scope))
                    visit(node.orelse, maybe(scope))
                else:
                    visit(node.orelse if checker_only else node.body, scope)
            elif isinstance(node, _TRY_NODES):
                visit(node.body, maybe(scope) if node.handlers else scope)
                for handler in node.handlers:
                    visit(handler.body, maybe(scope))
                visit(node.orelse, maybe(scope))
                visit(node.finalbody, scope)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
                visit(node.body, maybe(scope))
                visit(getattr(node, "orelse", []), maybe(scope))
            elif isinstance(node, ast.Match):
                for case in node.cases:
                    visit(case.body, maybe(scope))

    visit(tree.body, "top")
    return top, nested


def _repeated_imports(path: Path, package: str) -> List[Tuple[int, str]]:
    """``(line, module)`` of each function-level import of a module that
    *path* already imports whenever it loads."""
    top, nested = _top_and_function_imports(path, package)
    return [(line, target) for line, target in nested if target in top]


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


def _heavy(loaded: List[str]) -> List[str]:
    return [m for m in loaded if m == "yaml" or m.startswith("utils.providers")
            or m == "utils.provider_registry"]


def test_a_light_leaf_loads_no_provider_and_no_yaml():
    for name in LIGHT_LEAVES:
        code, stderr, loaded = _results()[name]
        assert code == 0, stderr
        assert _heavy(loaded) == [], f"importing {name} loads {_heavy(loaded)}"


def test_a_light_entry_script_loads_no_provider_and_no_yaml():
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        results = list(pool.map(_fresh_import, LIGHT_SCRIPTS))
    for name, code, stderr, loaded in results:
        assert code == 0, stderr
        assert _heavy(loaded) == [], f"importing {name} loads {_heavy(loaded)}"


def _skill_sources() -> List[Tuple[Path, str]]:
    """Every skill module outside ``tests/``, with the package it belongs to."""
    sources = []
    for path in sorted(SKILL_ROOT.rglob("*.py")):
        rel = path.relative_to(SKILL_ROOT)
        if any(part in ("tests", "__pycache__") or part.startswith(".") for part in rel.parts):
            continue
        parts = rel.with_suffix("").parts
        sources.append((path, ".".join(parts[:-1])))
    return sources


def test_the_executor_imports_no_module_twice():
    """A module imported at the top is not imported again inside a function:
    a second, lazy import of it guards no cycle and hides the real dependency
    (docs/development/dispatcher.md#import-safety)."""
    repeats = [
        f"utils/{name}.py:{line} {target}"
        for name in EXECUTOR_MODULES
        for line, target in _repeated_imports(UTILS / f"{name}.py", "utils")
    ]
    assert repeats == [], "\n".join(repeats)


def test_no_skill_module_imports_a_module_twice():
    """The same rule for every module of the skill: what a module imports at
    the top, it does not import again inside a function."""
    repeats = [
        f"{path.relative_to(SKILL_ROOT)}:{line} {target}"
        for path, package in _skill_sources()
        for line, target in _repeated_imports(path, package)
    ]
    assert repeats == [], "\n".join(repeats)


def _repeats_in(tmp_path: Path, source: str) -> List[str]:
    module = tmp_path / "fixture_module.py"
    module.write_text(source, encoding="utf-8")
    return [target for _line, target in _repeated_imports(module, "")]


@pytest.mark.parametrize("source, repeated", [
    pytest.param(
        "import sys\nif sys.platform == 'win32':\n    import ctypes\n\n"
        "def f():\n    import ctypes\n",
        [], id="platform-conditional-top"),
    pytest.param(
        "try:\n    import tomllib\nexcept ImportError:\n    tomllib = None\n\n"
        "def f():\n    import tomllib\n",
        [], id="try-except-importerror"),
    pytest.param(
        "from typing import TYPE_CHECKING as TC\nif TC:\n    import decimal\n\n"
        "def f():\n    import decimal\n",
        [], id="type-checking-alias"),
    pytest.param(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import decimal\n"
        "else:\n    import fractions\n\ndef f():\n    import fractions\n",
        ["fractions"], id="type-checking-else"),
    pytest.param(
        "from typing import TYPE_CHECKING as TC\nif TC:\n    pass\n"
        "else:\n    import fractions\n\ndef f():\n    import fractions\n",
        ["fractions"], id="type-checking-alias-else"),
    pytest.param(
        "import json\n\nclass A:\n    def f(self):\n        import json\n",
        ["json"], id="genuine-duplicate"),
])
def test_the_repeat_guard_counts_only_imports_every_load_runs(tmp_path, source, repeated):
    """A conditional top-level import (a platform test, ``except ImportError``,
    an ``if TYPE_CHECKING:`` body under any alias) is no dependency a lazy import
    repeats; the runtime ``else`` of ``if TYPE_CHECKING:`` is one."""
    assert _repeats_in(tmp_path, source) == repeated


# ---------------------------------------------------------------------------
# Reader paths: what a short-lived command loads when it runs, not only when
# its module is imported.
# ---------------------------------------------------------------------------

#: The provider adapters, which ``utils/providers/__init__.py`` loads together.
_ADAPTERS = "utils.providers.*"

#: Each commit front door's ``--help``: the front door's own module and the
#: provider registry it imports, nothing that resolves a range, writes a brief
#: or runs a phase.
_FRONT_DOOR_HELP = frozenset({
    "modes", "utils", "utils.atomic_io", "utils.cli_common", "utils.clock",
    "utils.commit_front_door", "utils.commit_paths", "utils.git_utils", "utils.held_folder",
    "utils.json_extractor", "utils.output_handler", "utils.process_control",
    "utils.process_state", "utils.provider_registry", "utils.providers", _ADAPTERS,
    "utils.readonly_audit", "utils.redact", "utils.retry_ladder", "utils.review_diff",
    "utils.scope_policy", "utils.status_file", "utils.stream_bootstrap", "utils.tasks_marker",
    "utils.text_clip", "utils.unattended", "utils.yaml_io",
})

#: The project modules each reader path loads (the set it loaded before the
#: function-level imports were hoisted, ef8d750e). ``status --view run`` of a
#: stopped review-then-apply composite spells its continuation through
#: ``improve_workflow``, and reads the workflow; it runs no review or apply, so
#: the review engine, the executor and the write path's reporting stay unloaded.
#: improve-commits' door also hands a worker its stage applier (``detach``).
#: A path may load fewer modules; a module missing here is one an import made
#: eager (docs/development/dispatcher.md#import-safety).
READER_PATHS: Dict[str, frozenset] = {
    "status": frozenset({
        "check_workflow_prerequisites", "full_workflow_orchestrator", "modes", "utils",
        "utils.advisory", "utils.atomic_io", "utils.batch_reply", "utils.check_records",
        "utils.cli_common", "utils.clock", "utils.codex_trust", "utils.commit_apply",
        "utils.commit_paths", "utils.commit_range", "utils.commit_spelling", "utils.detach",
        "utils.failure_capture", "utils.git_utils", "utils.held_folder", "utils.host_capabilities",
        "utils.human_decisions", "utils.importance", "utils.improve_workflow",
        "utils.json_extractor", "utils.judge_vocab", "utils.known_checks",
        "utils.landlock_dispatch", "utils.landlock_exec", "utils.llm_client", "utils.manual_tasks",
        "utils.metrics", "utils.next_phase", "utils.next_phase_words", "utils.output_handler",
        "utils.phase_walk", "utils.probe_state", "utils.process_capture", "utils.process_control",
        "utils.process_state", "utils.progress_tracker", "utils.prompt_channels",
        "utils.provider_outcome", "utils.provider_registry", "utils.providers", _ADAPTERS,
        "utils.readonly_audit", "utils.redact", "utils.result_files", "utils.retry_ladder",
        "utils.review_diff", "utils.routes", "utils.run_record", "utils.run_registry",
        "utils.run_status", "utils.runner_selection", "utils.sandbox_walk",
        "utils.schema_validator", "utils.scope_apply", "utils.scope_policy",
        "utils.scope_policy_runs", "utils.state_file", "utils.state_manager",
        "utils.status_commands", "utils.status_file", "utils.status_native", "utils.status_text",
        "utils.status_views", "utils.stream_bootstrap", "utils.structured_output",
        "utils.subagent_prompts", "utils.subagent_tasks", "utils.task_ledger", "utils.task_outcome",
        "utils.tasks_file", "utils.tasks_marker", "utils.text_clip", "utils.text_fmt",
        "utils.unattended", "utils.validation_context", "utils.write_safety", "utils.yaml_io",
    }),
    "review_commits_orchestrator.py": _FRONT_DOOR_HELP,
    "apply_commit_fixes_orchestrator.py": _FRONT_DOOR_HELP,
    "improve_commits_orchestrator.py": _FRONT_DOOR_HELP | {
        "utils.advisory", "utils.detach", "utils.run_record", "utils.run_registry",
    },
}

#: argv: the skill root, then ``status <plan>`` or ``<entry script>`` (run
#: with ``--help``). Prints the exit, the stdout and the project modules loaded.
_READER_PROBE = r'''
import contextlib, io, json, runpy, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
out, code = io.StringIO(), None
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
    if sys.argv[2] == "status":
        from utils import run_status

        code = run_status.main(["--plan-file", sys.argv[3], "--view", "run"])
    else:
        sys.argv = [str(root / sys.argv[2]), "--help"]
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            code = exc.code


def mine(module):
    path = getattr(module, "__file__", None)
    if not path:
        return False
    path = Path(path).resolve()
    try:
        first = path.relative_to(root).parts[0]
    except ValueError:
        return False
    return not first.startswith(".")


print(json.dumps({"code": code, "out": out.getvalue(),
                  "loaded": sorted(n for n, m in list(sys.modules.items()) if mine(m))}))
'''


def _run_reader(path: str, repo: Path, plan: Path) -> Tuple[str, Dict]:
    env = {k: v for k, v in os.environ.items()
           if k != "PYTHONPATH" and not k.startswith("MULTI_LLM_")}
    argv = [sys.executable, "-c", _READER_PROBE, str(SKILL_ROOT), path]
    if path == "status":
        argv.append(str(plan))
    done = subprocess.run(argv, cwd=repo, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return path, json.loads(done.stdout.strip().splitlines()[-1])


def test_a_reader_path_loads_no_module_its_base_did_not(tmp_path):
    """A stopped improve-code run's ``status --view run`` and each commit front
    door's ``--help`` run in a fresh interpreter, in a git repository; the
    project modules they load stay within :data:`READER_PATHS`."""
    from utils import status_file

    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    plan = repo / "plan.md"
    plan.write_text("# plan\n", encoding="utf-8")
    phase = repo / "plan" / "improve-code"
    phase.mkdir(parents=True)
    status_file.claim_run(str(phase), {
        "run_id": "run-d2-000001", "state": "failed", "exit_code": 3,
        "error": "needs_human (exit 3)", "detached": True, "mode": "improve-code",
        "replay_flags": ["--force", "--files", "a.py", "--detach"],
    })
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as pool:
        results = dict(pool.map(lambda path: _run_reader(path, repo, plan), READER_PATHS))

    status = results["status"]
    assert status["code"] == 3, status["out"]
    resume = [line for line in status["out"].splitlines() if "resume with:" in line]
    assert len(resume) == 1 and resume[0].endswith("--files a.py --resume"), status["out"]
    assert "utils.improve_workflow" in status["loaded"]
    grown = {}
    for path, result in results.items():
        if path != "status":
            assert result["code"] == 0 and "usage:" in result["out"], (path, result["out"])
            assert "utils.commit_front_door" in result["loaded"], path
        loaded = {_ADAPTERS if name.startswith("utils.providers.") else name
                  for name in result["loaded"]}
        added = sorted(loaded - READER_PATHS[path])
        if added:
            grown[path] = added
    assert grown == {}, (
        "\n".join(f"{path} now loads {added}" for path, added in grown.items())
        + "\nkeep those imports in the functions that need them, or list the "
        "modules in READER_PATHS if the path needs them")
