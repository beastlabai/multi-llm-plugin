"""Import layers: a ratchet that stops new upward imports.

Every skill module (everything outside ``tests/``) belongs to one layer of
``LAYERS``, and an import may point only at its own layer or a lower one:

    L0 core           pure helpers: text, JSON, schemas, retry words, mode table
    L1 state          run state and protocol: status files, git, CLI plumbing
    L2 providers      provider adapters, the registry, model invocation
    L3 execution      subagent execution, write scheduling and safety, detach
    L4 domain         review, apply and commit logic shared by orchestrators
    L5 orchestrators  one module per mode, plus the workflow scripts
    L6 CLI            the dispatcher and the user-facing utility commands

``KNOWN_UPWARD`` lists the upward edges that existed when the map was drawn.
It may only shrink: a new upward edge fails ``test_no_new_upward_import``, and
an entry whose edge no longer points upward fails
``test_known_upward_edges_still_point_upward`` until it is deleted.

An edge is "module A imports module B" at least once, wherever the import sits:
module level, inside a function or class, or in either branch of a
``try/except ImportError`` shim. Spellings are normalised, so ``from .x``,
``from utils.x``, ``import utils.x`` and a shim's bare ``from x`` all name
``utils.x``. ``from package import name`` follows Python's lookup: a name the
package's ``__init__`` binds wins, and resolves through every re-export and
alias to the module that defines it; only an unbound name means the submodule.
A wildcard import of a skill module fails
``test_no_wildcard_import_of_a_skill_module``.

Dynamic imports are calls of ``importlib.import_module``, ``__import__`` and
``next_phase._skill_module``, or of an alias of one, found in the syntax tree
under the same rules, so comments and strings never make an edge. A call with
literal arguments is an edge like any import, relative
``import_module(".x", package=...)`` and ``__import__(..., level=1)``
included. A call whose module name is computed, or a loader used as a value,
needs a ``COMPUTED_IMPORTS`` entry naming the modules it can load, which then
count as edges; ``test_computed_imports_are_declared`` fails without one.

Not counted: imports under ``if TYPE_CHECKING:`` and the package ``__init__``
Python runs implicitly.

What to do when a check fails: docs/development/dispatcher.md#import-layers.
"""

from __future__ import annotations

import ast
import fnmatch
import functools
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
_SKIPPED_PARTS = {"tests", "__pycache__", ".venv"}

#: Layer number -> (name, members). Members are dotted module names relative to
#: the skill directory; ``*`` patterns cover whole families, and an exact name
#: beats a pattern. Every module must match exactly one layer.
LAYERS: Dict[int, Tuple[str, Tuple[str, ...]]] = {
    0: ("core", (
        "modes",
        "utils.advisory",
        "utils.env_vars",
        "utils.importance",
        "utils.json_extractor",
        "utils.judge_vocab",
        "utils.launch_limiter",
        "utils.phase_walk",
        "utils.process_capture",
        "utils.process_control",
        "utils.process_state",
        "utils.prompt_files",
        "utils.providers.result_envelope",
        "utils.report_markdown",
        "utils.retry_ladder",
        "utils.schema_validator",
        "utils.scope_policy",
        "utils.stream_bootstrap",
        "utils.text_clip",
        "utils.yaml_io",
    )),
    1: ("state", (
        "utils.cli_common",
        "utils.cli_version_cache",
        "utils.git_diff",
        "utils.git_utils",
        "utils.host_capabilities",
        "utils.host_paths",
        "utils.host_routes",
        "utils.host_routes.*",
        "utils.host_shim",
        "utils.metrics",
        "utils.output_handler",
        "utils.review_diff",
        "utils.run_registry",
        "utils.state_manager",
        "utils.status_file",
        "utils.task_ledger",
        "utils.unattended",
    )),
    2: ("providers", (
        "utils.codex_trust",
        "utils.landlock_dispatch",
        "utils.landlock_exec",
        "utils.llm_client",
        "utils.provider_outcome",
        "utils.provider_registry",
        "utils.providers",
        "utils.providers.*",
        "utils.readonly_audit",
        "utils.result_files",
        "utils.structured_output",
    )),
    3: ("execution", (
        "utils.batch_done",
        "utils.batch_prompt",
        "utils.check_records",
        "utils.check_reply",
        "utils.check_tree",
        "utils.code_fix_batcher",
        "utils.commit_paths",
        "utils.commit_range",
        "utils.commit_write_guard",
        "utils.detach",
        "utils.human_decisions",
        "utils.human_task_plan",
        "utils.known_checks",
        "utils.plan_updater",
        "utils.prior_changes",
        "utils.progress_tracker",
        "utils.report_check",
        "utils.restore_batch",
        "utils.scope_apply",
        "utils.stop_answers",
        "utils.subagent_executor",
        "utils.subagent_prompts",
        "utils.subagent_tasks",
        "utils.suggestion_batcher",
        "utils.task_decomposer",
        "utils.tasks_file",
        "utils.validation_batcher",
        "utils.verification_lines",
        "utils.wakeup_check",
        "utils.write_batch_scheduler",
        "utils.write_phase_bridge",
        "utils.write_prompt",
        "utils.write_record",
        "utils.write_safety",
        "utils.write_verdict",
    )),
    4: ("domain", (
        "utils",  # the re-export barrel
        "utils.apply_orchestrator_base",
        "utils.apply_output_helpers",
        "utils.apply_path_helpers",
        "utils.apply_report",
        "utils.apply_selection_helpers",
        "utils.apply_suggestion_phase",
        "utils.backup",
        "utils.commit_apply",
        "utils.commit_attribution",
        "utils.commit_report",
        "utils.commit_spelling",
        "utils.consolidation",
        "utils.file_discovery",
        "utils.filtering",
        "utils.html_report_generator",
        "utils.interactive",
        "utils.prompt_loader",
        "utils.report_parser",
        "utils.report_view",
        "utils.review_orchestrator_base",
        "utils.scope_policy_runs",
        "utils.scope_report",
        "utils.scope_round",
        "utils.selection_migration",
        "utils.slot_guard",
        "utils.suggestion_processor",
        "utils.tasks_graph_check",
        "utils.validation",
        "utils.validation_context",
        "utils.validation_sections",
    )),
    5: ("orchestrators", (
        "*_orchestrator",
        "check_workflow_prerequisites",
        "display_decisions",
        "finalize_tracking",
        "update_plan_tasks",
        "utils.commit_brief",
        "utils.commit_front_door",
        "utils.commit_review",
        "utils.improve_workflow",
    )),
    6: ("CLI", (
        "hosts.render_capabilities",
        "init_config",
        "multi_llm",
        "utils.doctor",
        "utils.host_install",
        "utils.improve_steps",
        "utils.next_phase",
        "utils.resume_state",
        "utils.run_status",
        "utils.status_commands",
        "utils.status_native",
        "utils.status_text",
        "utils.status_views",
    )),
}

#: (importer, target) -> what crosses the edge and where it belongs instead.
#: Delete an entry once its edge is gone; never add one without a reason.
KNOWN_UPWARD: Dict[Tuple[str, str], str] = {
    # Redaction and the state home live in the 2,500-line write_safety (L3).
    ("utils.landlock_dispatch", "utils.write_safety"):
        "state_home: belongs in output_handler (L1)",
    ("utils.provider_outcome", "utils.write_safety"):
        "redact_text: belongs in a leaf redaction module that write_safety re-imports",
    ("utils.providers.codex_stream_audit", "utils.write_safety"):
        "redact_text: belongs in a leaf redaction module that write_safety re-imports",
    ("utils.providers.cursor_agent", "utils.write_safety"):
        "redact_text: belongs in a leaf redaction module that write_safety re-imports",
    # Small helpers and constants in the wrong house.
    ("utils.subagent_executor", "utils.review_orchestrator_base"):
        "get_backoff_delay, run_tasks_concurrent: belong in a concurrency module at L3",
    ("utils.provider_registry", "utils.run_status"):
        "WAIT_POLL_INTERVAL_S: belongs in status_file (L1)",
    ("utils.output_handler", "utils.plan_updater"):
        "get_tasks_file_path: belongs in output_handler itself",
    ("utils.result_files", "utils.progress_tracker"):
        "STATE_DONE, TERMINAL_STATES: belong in status_file (L1)",
    ("utils.llm_client", "utils.subagent_executor"):
        "_VERSION_CACHE, _PROBE_VERSION_CACHE: belong in providers.base",
    ("utils.providers.result_envelope", "utils.providers.base"):
        "STATUS_* are defined in retry_ladder (L0); the two CATEGORY_* constants belong there too",
    ("utils.commit_front_door", "multi_llm"):
        "_split_positionals, _revs_flags: belong in modes or cli_common",
    # The run record lives in detach (L3).
    ("utils.result_files", "utils.detach"):
        "report_progress: belongs in a run-record module at L1",
    ("utils.unattended", "utils.detach"):
        "record_resolution: belongs in a run-record module at L1",
    ("utils.run_registry", "utils.detach"):
        "close_out_dead_run: the sweep should take it as an argument from its CLI",
    ("utils.detach", "utils.consolidation"):
        "consolidation_is_current: a hash comparison that belongs in output_handler (L1)",
    # next_phase (L6) also holds words the lower layers need.
    ("utils.cli_common", "utils.next_phase"):
        "recorded_plan_spelling: belongs in a routes module at L1, with resolve_route",
    ("utils.detach", "utils.next_phase"):
        "resolve_route, _skill_module: belong in a routes module at L1",
    ("utils.write_batch_scheduler", "utils.next_phase"):
        "manual_close_note, manual_gate_why, manual_gate_decline_answers: "
        "belong in a manual-tasks module at L3",
    ("utils.stop_answers", "utils.next_phase"):
        "MAX_RECORDED_WHY_CHARS: belongs in a manual-tasks module at L3",
    ("utils.apply_orchestrator_base", "utils.next_phase"):
        "resolve_route, phase_command: belong in routes (L1) and a command-words module at L4",
    ("implement_orchestrator", "utils.next_phase"):
        "after_code_note, close_command, manual_close_note, phase_command, resolve_route: "
        "belong in routes, manual-tasks and command-words modules",
    ("full_workflow_orchestrator", "utils.next_phase"):
        "commit_mode_word, commit_view_for: belong in a command-words module at L4",
    # The scheduler reaches up for tracking, its commit-guard helpers for the guard.
    ("utils.write_batch_scheduler", "finalize_tracking"):
        "the tracking body: belongs in a utils module; the script stays a thin CLI",
    ("utils.commit_write_guard", "utils.commit_apply"):
        "save_guard, recheck_files, repin_after_batch and the rest of the commit guard: "
        "belong in this commit-guard module at L3",
    # The front door (L5) also holds the stage a detached worker applies.
    ("utils.detach", "utils.commit_front_door"):
        "apply_stage, stage_takes_over: belong in a commit-stage module at L3",
    ("utils.llm_client", "utils.commit_range"):
        "pinned_head_read_rule: belongs in a commit-paths module, or becomes a parameter",
}


#: Dynamic imports whose module name is computed: (importer, the call as
#: ``ast.unparse`` prints it) -> (the modules it can load, why the name is
#: computed). Targets are module names or ``*`` patterns, and each counts as an
#: edge of the importer, so the layer checks see them. Delete an entry once its
#: call is gone; a new computed call fails until it has one.
COMPUTED_IMPORTS: Dict[Tuple[str, str], Tuple[Tuple[str, ...], str]] = {
    ("multi_llm", "importlib.import_module(module_path)"): (
        ("*_orchestrator", "init_config", "utils.doctor", "utils.host_install",
         "utils.run_status"),
        "the dispatcher loads each mode's target_module from the modes table",
    ),
    ("full_workflow_orchestrator", "importlib.import_module(mode.target_module)"): (
        ("*_orchestrator",),
        "run_mode loads the orchestrator of each workflow step from the modes table",
    ),
    ("utils.next_phase", "importlib.import_module(name)"): (
        (),
        "the body of _skill_module, whose callers pass literal names: each "
        "_skill_module(\"x\") call is an edge of its caller",
    ),
    ("utils.providers", "importlib.import_module(f'{__name__}.{info.name}')"): (
        ("utils.providers.*",),
        "adapter discovery imports every module of the package",
    ),
}


# ---------------------------------------------------------------------------
# Building the graph
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    """One import of a target: where it is and the name it brings in, or the
    loader call for a dynamic import."""

    path: str
    line: int
    name: str
    in_function: bool
    dynamic: bool = False

    def render(self) -> str:
        verb = "calls" if self.dynamic else "imports"
        where = " (inside a function)" if self.in_function else ""
        return f"{self.path}:{self.line} {verb} {self.name}{where}"


@dataclass(frozen=True)
class Graph:
    """What ``build_graph`` read from one skill tree."""

    modules: Tuple[str, ...]
    #: (importer, target) -> the sites that import it.
    edges: Dict[Tuple[str, str], Tuple[Site, ...]]
    #: (importer, call) -> the sites of a dynamic import whose module name is
    #: computed, declared in COMPUTED_IMPORTS or not.
    computed: Dict[Tuple[str, str], Tuple[Site, ...]]
    #: ``from <skill module> import *`` statements.
    wildcards: Tuple[Site, ...]


#: Callables that import the module a string names. ``_skill_module`` is
#: ``utils/next_phase.py``'s wrapper around ``importlib.import_module``.
_LOADERS = ("import_module", "__import__", "_skill_module")
_PARAMETERS = {
    "import_module": ("name", "package"),
    "__import__": ("name", "globals", "locals", "fromlist", "level"),
    "_skill_module": ("name",),
}
_STATEMENT_FIELDS = ("body", "orelse", "finalbody", "handlers", "cases")
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPUTED = object()  # an argument that is not a literal


def _module_files(root: Path) -> Dict[str, Tuple[Path, bool]]:
    """Dotted module name -> (path, is a package ``__init__``)."""
    files: Dict[str, Tuple[Path, bool]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(p in _SKIPPED_PARTS or p.startswith(".") for p in rel.parts):
            continue
        parts = rel.with_suffix("").parts
        is_package = parts[-1] == "__init__"
        files[".".join(parts[:-1] if is_package else parts)] = (path, is_package)
    return files


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _loader(node: ast.AST, loaders: Mapping[str, str]) -> Optional[str]:
    """The loader *node* names (``import_module``, ``x.import_module``, an alias)."""
    if isinstance(node, ast.Name):
        return loaders.get(node.id)
    if isinstance(node, ast.Attribute) and node.attr in _LOADERS:
        return node.attr
    return None


def _walk(body: Sequence[ast.stmt], loaders: Optional[Mapping[str, str]] = None,
          ) -> Iterator[Tuple[ast.AST, bool]]:
    """(node, inside a function) for every import statement at any depth, and,
    given *loaders* (local name -> loader), every call or other use of a loader.

    One scope rule for both: the body of ``if TYPE_CHECKING:`` never runs, so it
    is skipped. Without *loaders* only statements are walked, which is what keeps
    the whole graph under a second; expressions are walked only in modules that
    spell a loader's name, the only ones that can call one.
    """
    stack: List[Tuple[ast.AST, bool]] = [(node, False) for node in reversed(body)]
    while stack:
        node, in_function = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, in_function
            continue
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend((child, in_function) for child in reversed(node.orelse))
            continue
        if loaders is None:
            fields: Sequence[str] = _STATEMENT_FIELDS
        else:
            if isinstance(node, ast.Call) and _loader(node.func, loaders):
                yield node, in_function
                # The receiver and the arguments may hold more calls; the
                # loader's own name is not a second use of it.
                func = node.func
                inner = ([func.value] if isinstance(func, ast.Attribute) else [])
                inner += [*node.args, *(k.value for k in node.keywords)]
                stack.extend((child, in_function) for child in reversed(inner))
                continue
            if (isinstance(node, (ast.Name, ast.Attribute))
                    and isinstance(node.ctx, ast.Load) and _loader(node, loaders)):
                yield node, in_function
                continue
            fields = node._fields
        nested = in_function or isinstance(node, _FUNCTIONS)
        for field in reversed(fields):
            value = getattr(node, field, None)
            scope = nested if field == "body" else in_function
            if isinstance(value, list):
                stack.extend((child, scope) for child in reversed(value)
                             if isinstance(child, ast.AST))
            elif isinstance(value, ast.AST) and not isinstance(value, ast.expr_context):
                stack.append((value, scope))


def _top_level(body: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """The statements that run when a module is imported: every branch, but not
    function or class bodies, nor ``if TYPE_CHECKING:``."""
    for node in body:
        yield node
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            yield from _top_level(node.orelse)
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            yield from _top_level(getattr(node, field, None) or ())
        for block in (*(getattr(node, "handlers", None) or ()),
                      *(getattr(node, "cases", None) or ())):
            yield from _top_level(block.body)


def _literal(node: Optional[ast.expr], default: object = None) -> object:
    if node is None:
        return default
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _COMPUTED


#: A name a package binds: (the module it comes from, its name there). The
#: module is None when the name is defined in the package itself or comes from
#: outside the skill; the name is None when the binding is that module itself.
Binding = Tuple[Optional[str], Optional[str]]


class _Resolver:
    def __init__(self, files: Dict[str, Tuple[Path, bool]],
                 packages: Dict[str, ast.Module]) -> None:
        self.files = files
        #: Package -> {name its ``__init__`` binds: every binding of that name}.
        self.bindings: Dict[str, Dict[str, List[Binding]]] = {
            module: self._bindings(module, tree.body) for module, tree in packages.items()
        }

    def _bindings(self, package: str, body: Sequence[ast.stmt]) -> Dict[str, List[Binding]]:
        names: Dict[str, List[Binding]] = {}
        for node in _top_level(body):
            bound: List[Tuple[str, Binding]] = []
            if isinstance(node, ast.ImportFrom):
                base = self.base(package, node)
                bound = [(a.asname or a.name, (base, a.name if base else None))
                         for a in node.names if a.name != "*"]
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        bound.append((a.asname, (self.imported(package, a.name), None)))
                    else:
                        top = a.name.partition(".")[0]
                        bound.append((top, (self.absolute(package, top), None)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound = [(node.name, (None, None))]
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                bound = [(n.id, (None, None)) for t in targets for n in ast.walk(t)
                         if isinstance(n, ast.Name)]
            for name, binding in bound:
                names.setdefault(name, []).append(binding)
        return names

    def package_of(self, module: str) -> str:
        return module if self.files[module][1] else module.rpartition(".")[0]

    def absolute(self, importer: str, name: str) -> Optional[str]:
        """*name* as the skill module it means, trying the bare sibling
        spelling of a dual-import shim when *name* is not a skill module."""
        if name in self.files:
            return name
        package = self.package_of(importer)
        while package:
            if f"{package}.{name}" in self.files:
                return f"{package}.{name}"
            package = package.rpartition(".")[0]
        return None

    def imported(self, importer: str, dotted: str) -> Optional[str]:
        """``import dotted``: the deepest skill module the name reaches."""
        parts = dotted.split(".")
        for end in range(len(parts), 0, -1):
            target = self.absolute(importer, ".".join(parts[:end]))
            if target is not None:
                return target
        return None

    @staticmethod
    def relative(package: str, level: int, name: str) -> Optional[str]:
        """What *level* leading dots and *name* mean inside *package*."""
        parts = package.split(".") if package else []
        if level - 1 >= len(parts):
            return None  # above the top-level package
        return ".".join(parts[: len(parts) - (level - 1)] + ([name] if name else []))

    def base(self, importer: str, node: ast.ImportFrom) -> Optional[str]:
        """The module a ``from ... import`` statement reads from."""
        if not node.level:
            return self.absolute(importer, node.module or "")
        dotted = self.relative(self.package_of(importer), node.level, node.module or "")
        return dotted if dotted in self.files else None

    def resolve_from(self, base: str, name: str,
                     seen: Tuple[Tuple[str, str], ...] = ()) -> Tuple[str, ...]:
        """``from base import name``: the modules that define *name*.

        A plain module is the target itself. A package is looked up as Python
        does: a name its ``__init__`` binds first, followed through the binding's
        original name (so an alias of a re-export of a re-export ends at the
        defining module), and the submodule only when nothing binds the name.
        """
        if base not in self.bindings:
            return (base,)
        found: List[str] = []
        if (base, name) not in seen:
            for source, original in self.bindings[base].get(name, ()):
                if source is None:
                    found.append(base)
                elif original is None:
                    found.append(source)
                else:
                    found.extend(self.resolve_from(source, original, seen + ((base, name),)))
        if found:
            return tuple(dict.fromkeys(found))
        submodule = f"{base}.{name}"
        return (submodule,) if submodule in self.files else (base,)

    def dynamic(self, importer: str, call: ast.Call, loader: str) -> Optional[Tuple[str, ...]]:
        """The skill modules a loader call imports (empty when it names none),
        or None when its arguments are not literals."""
        params = _PARAMETERS[loader]
        if (len(call.args) > len(params)
                or any(isinstance(a, ast.Starred) for a in call.args)
                or any(k.arg is None for k in call.keywords)):
            return None
        given: Dict[str, ast.expr] = dict(zip(params, call.args))
        given.update((k.arg, k.value) for k in call.keywords if k.arg)
        name = _literal(given.get("name"))
        if not isinstance(name, str):
            return None
        fromlist: object = ()
        if loader == "__import__":
            level = _literal(given.get("level"), 0)
            fromlist = _literal(given.get("fromlist"), ()) or ()
            if (not isinstance(level, int) or level < 0
                    or not isinstance(fromlist, (list, tuple))
                    or not all(isinstance(n, str) for n in fromlist)):
                return None
            dotted = self.relative(self.package_of(importer), level, name) if level else name
        elif loader == "import_module" and name.startswith("."):
            package = self._package_argument(importer, given.get("package"))
            if package is None:
                return None
            level = len(name) - len(name.lstrip("."))
            dotted = self.relative(package, level, name[level:])
        else:
            dotted = name
        if dotted is None:
            return ()
        base = self.absolute(importer, dotted)
        if fromlist and base is not None:
            return tuple(dict.fromkeys(t for n in fromlist for t in self.resolve_from(base, n)))
        target = base or self.imported(importer, dotted)
        return () if target is None else (target,)

    def _package_argument(self, importer: str, node: Optional[ast.expr]) -> Optional[str]:
        """``import_module``'s *package*: a literal, ``__package__`` or ``__name__``."""
        if isinstance(node, ast.Name) and node.id == "__package__":
            return self.package_of(importer)
        if isinstance(node, ast.Name) and node.id == "__name__":
            return importer
        value = _literal(node)
        return value if isinstance(value, str) and value else None


def _loader_names(imports: Iterable[ast.AST]) -> Dict[str, str]:
    """Local name -> loader: the loaders' own names, plus any alias a module
    imports one under (``from importlib import import_module as load``)."""
    names = {name: name for name in _LOADERS}
    for node in imports:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _LOADERS and alias.asname:
                    names[alias.asname] = alias.name
    return names


def _matching(modules: Sequence[str], members: Iterable[str]) -> List[str]:
    """The modules that *members* (names or ``*`` patterns) name."""
    return [m for m in modules if any(fnmatch.fnmatchcase(m, member) for member in members)]


def build_graph(root: Path,
                computed_imports: Mapping[Tuple[str, str], Tuple[Tuple[str, ...], str]]
                = COMPUTED_IMPORTS) -> Graph:
    """Every import between the skill modules under *root*."""
    files = _module_files(root)
    # Only the package trees outlive their module's turn: holding all of them
    # makes every garbage collection walk the whole skill, doubling the time.
    packages = {module: ast.parse(path.read_bytes(), str(path))
                for module, (path, is_package) in files.items() if is_package}
    resolver = _Resolver(files, packages)
    modules = tuple(sorted(files))
    edges: Dict[Tuple[str, str], List[Site]] = {}
    computed: Dict[Tuple[str, str], List[Site]] = {}
    wildcards: List[Site] = []
    for module, (path, _) in files.items():
        source = path.read_bytes()
        tree = packages.get(module) or ast.parse(source, str(path))
        rel = path.relative_to(root).as_posix()
        loaders = None
        if any(name.encode() in source for name in _LOADERS):
            loaders = _loader_names(node for node, _ in _walk(tree.body))
        found: List[Tuple[str, Site]] = []
        for node, in_function in _walk(tree.body, loaders):
            line = getattr(node, "lineno", 0)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = resolver.imported(module, alias.name)
                    if target is not None:
                        found.append((target, Site(rel, line, alias.name, in_function)))
            elif isinstance(node, ast.ImportFrom):
                base = resolver.base(module, node)
                if base is None:
                    continue
                for alias in node.names:
                    site = Site(rel, line, alias.name, in_function)
                    if alias.name == "*":
                        wildcards.append(Site(rel, line, f"from {base} import *", in_function))
                        found.append((base, site))
                    else:
                        found.extend((t, site) for t in resolver.resolve_from(base, alias.name))
            else:
                assert loaders is not None
                site = Site(rel, line, ast.unparse(node), in_function, dynamic=True)
                targets: Optional[Tuple[str, ...]] = None
                loader = _loader(node.func, loaders) if isinstance(node, ast.Call) else None
                if loader is not None:
                    targets = resolver.dynamic(module, node, loader)
                if targets is None:
                    computed.setdefault((module, site.name), []).append(site)
                else:
                    found.extend((target, site) for target in targets)
        for target, site in found:
            if target != module:
                edges.setdefault((module, target), []).append(site)
    for (importer, call), (declared, _) in computed_imports.items():
        for site in computed.get((importer, call), ()):
            for target in _matching(modules, declared):
                if target != importer:
                    edges.setdefault((importer, target), []).append(site)

    def ordered(sites: List[Site]) -> Tuple[Site, ...]:
        return tuple(sorted(set(sites), key=lambda s: (s.path, s.line, s.name)))

    return Graph(
        modules=modules,
        edges={edge: ordered(sites) for edge, sites in sorted(edges.items())},
        computed={key: ordered(sites) for key, sites in sorted(computed.items())},
        wildcards=ordered(wildcards),
    )


@functools.lru_cache(maxsize=1)
def import_graph() -> Graph:
    return build_graph(SKILL_ROOT)


# ---------------------------------------------------------------------------
# Layer lookup
# ---------------------------------------------------------------------------

def _is_pattern(member: str) -> bool:
    return any(ch in member for ch in "*?[")


def layer_matches(module: str) -> List[int]:
    """The layers claiming *module*: exact entries first, else patterns."""
    exact = [n for n, (_, members) in LAYERS.items() if module in members]
    if exact:
        return exact
    return [
        n for n, (_, members) in LAYERS.items()
        if any(_is_pattern(m) and fnmatch.fnmatchcase(module, m) for m in members)
    ]


def layer_of(module: str) -> Optional[int]:
    matches = layer_matches(module)
    return matches[0] if len(matches) == 1 else None


def _label(layer: Optional[int]) -> str:
    return "no layer" if layer is None else f"L{layer} {LAYERS[layer][0]}"


# ---------------------------------------------------------------------------
# The checks, as functions of a graph so synthetic trees can run them too
# ---------------------------------------------------------------------------

def layer_problems(graph: Graph) -> List[str]:
    problems = []
    for module in graph.modules:
        matches = layer_matches(module)
        if len(matches) == 1:
            continue
        if matches:
            problems.append(f"{module} is listed in more than one layer: "
                            + ", ".join(_label(n) for n in matches))
            continue
        needs = [layer_of(t) for (a, t) in graph.edges if a == module]
        floor = max([n for n in needs if n is not None], default=0)
        users = [layer_of(a) for (a, t) in graph.edges if t == module]
        ceiling = min([n for n in users if n is not None], default=max(LAYERS))
        problems.append(
            f"{module} has no layer. Add it to LAYERS at the layer that fits its "
            f"role, from L{floor} (its imports need at least that) to L{ceiling} "
            f"(its importers allow at most that)"
        )
    for number, (_, members) in LAYERS.items():
        for member in members:
            if not _matching(graph.modules, (member,)):
                problems.append(f"LAYERS L{number} names {member!r}, which matches no "
                                "module; delete or rename the entry")
    return problems


def upward_problems(graph: Graph,
                    known: Mapping[Tuple[str, str], str] = KNOWN_UPWARD) -> List[str]:
    blocks = []
    for (importer, target), sites in graph.edges.items():
        low, high = layer_of(importer), layer_of(target)
        if low is None or high is None or low >= high or (importer, target) in known:
            continue
        names = sorted({site.name for site in sites if not site.dynamic})
        crossing = ", ".join(names) or f"what it uses from {target}"
        blocks.append(
            f"{importer} ({_label(low)}) imports {target} ({_label(high)}):\n      "
            + "\n      ".join(site.render() for site in sites)
            + f"\n    Fix: move {crossing} down to L{low} or lower (the old "
            f"module can re-import it, so other callers keep working), or pass it "
            f"in as an argument. If the edge is deliberate, add\n"
            f"      (\"{importer}\", \"{target}\"): \"<what crosses and why it must point up>\",\n"
            f"    to KNOWN_UPWARD in tests/test_import_layers.py."
        )
    return blocks


def stale_problems(graph: Graph,
                   known: Mapping[Tuple[str, str], str] = KNOWN_UPWARD) -> List[str]:
    stale = []
    for (importer, target), reason in known.items():
        if not reason.strip():
            stale.append(f"({importer!r}, {target!r}) has no reason; give it one")
        elif (importer, target) not in graph.edges:
            stale.append(f"({importer!r}, {target!r}): the import is gone")
        else:
            low, high = layer_of(importer), layer_of(target)
            if low is not None and high is not None and low >= high:
                stale.append(f"({importer!r}, {target!r}): now {_label(low)} -> "
                             f"{_label(high)}, no longer upward")
    return stale


def computed_problems(graph: Graph,
                      declared: Mapping[Tuple[str, str], Tuple[Tuple[str, ...], str]]
                      = COMPUTED_IMPORTS) -> List[str]:
    problems = []
    for (importer, call), sites in graph.computed.items():
        if (importer, call) in declared:
            continue
        problems.append(
            f"{importer} imports a module whose name is computed:\n      "
            + "\n      ".join(site.render() for site in sites)
            + f"\n    Fix: import it by a literal name, or add\n"
            f"      ({importer!r}, {call!r}): ((\"<module or pattern>\", ...), "
            f"\"<why the name is computed>\"),\n"
            f"    to COMPUTED_IMPORTS in tests/test_import_layers.py, naming every "
            f"module it can load; they count as imports of {importer}."
        )
    for (importer, call), (targets, why) in declared.items():
        entry = f"COMPUTED_IMPORTS ({importer!r}, {call!r})"
        if (importer, call) not in graph.computed:
            problems.append(f"{entry}: the call is gone; delete the entry")
        if not why.strip():
            problems.append(f"{entry} has no reason; give it one")
        for member in targets:
            if not _matching(graph.modules, (member,)):
                problems.append(f"{entry} names {member!r}, which matches no module")
    return problems


def wildcard_problems(graph: Graph) -> List[str]:
    return [
        f"{site.path}:{site.line} {site.name}: name the imports instead, so each "
        f"edge points at the module that defines the name"
        for site in graph.wildcards
    ]


def test_every_module_has_exactly_one_layer():
    problems = layer_problems(import_graph())
    assert not problems, (
        "Every skill module belongs to exactly one layer "
        "(LAYERS in tests/test_import_layers.py):\n  " + "\n  ".join(problems)
    )


def test_no_new_upward_import():
    blocks = upward_problems(import_graph())
    assert not blocks, (
        "New upward import: a module may import only its own layer or a lower one "
        "(layer map in tests/test_import_layers.py, see "
        "docs/development/dispatcher.md#import-layers).\n  " + "\n  ".join(blocks)
    )


def test_known_upward_edges_still_point_upward():
    stale = stale_problems(import_graph())
    assert not stale, (
        "KNOWN_UPWARD only shrinks: delete each entry whose edge is gone or no "
        "longer upward, and give every entry a reason (tests/test_import_layers.py):"
        "\n  " + "\n  ".join(stale)
    )


def test_computed_imports_are_declared():
    problems = computed_problems(import_graph())
    assert not problems, (
        "A dynamic import with a computed module name must declare what it can "
        "load (COMPUTED_IMPORTS in tests/test_import_layers.py, see "
        "docs/development/dispatcher.md#import-layers):\n  " + "\n  ".join(problems)
    )


def test_no_wildcard_import_of_a_skill_module():
    problems = wildcard_problems(import_graph())
    assert not problems, (
        "A wildcard import of a skill module hides which module each name comes "
        "from (tests/test_import_layers.py):\n  " + "\n  ".join(problems)
    )


# ---------------------------------------------------------------------------
# The graph builder on synthetic trees
# ---------------------------------------------------------------------------

#: A minimal skill tree; each case replaces or adds files. The names are real
#: modules, so LAYERS places them: text_clip and modes L0, output_handler L1,
#: providers L2, write_safety and plan_updater L3.
_TREE = {
    "modes.py": "",
    "utils/__init__.py": "",
    "utils/next_phase.py": "def _skill_module(name):\n    pass\n",
    "utils/output_handler.py": "",
    "utils/plan_updater.py": "def get_tasks_file_path():\n    pass\n",
    "utils/providers/__init__.py": "",
    "utils/providers/result_envelope.py": "class ResultEnvelope:\n    pass\n",
    "utils/text_clip.py": "",
    "utils/write_safety.py": "def redact_text(text):\n    return text\n",
}
_UP = ("utils.text_clip", "utils.write_safety")  # L0 -> L3


def _graph(root: Path, files: Mapping[str, str]) -> Graph:
    """Write ``_TREE`` with *files* over it, and read it with nothing declared."""
    for rel, text in {**_TREE, **files}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return build_graph(root, {})


def _in_text_clip(statements: str) -> Dict[str, str]:
    return {"utils/text_clip.py": "import importlib\n\n\ndef clip():\n"
            + textwrap.indent(textwrap.dedent(statements), "    ")}


@pytest.mark.parametrize("call", [
    'importlib.import_module("utils.write_safety")',
    'importlib.import_module("write_safety")',
    'importlib.import_module(".write_safety", package="utils")',
    'importlib.import_module(".write_safety", __package__)',
    'importlib.import_module("..write_safety", package="utils.providers")',
    '__import__("utils.write_safety")',
    '__import__("utils.write_safety", fromlist=["redact_text"])',
    '__import__("utils", fromlist=["write_safety"])',
    '__import__("write_safety", globals(), None, ["redact_text"], 1)',
    '__import__("", globals(), None, ("write_safety",), level=1)',
    '_skill_module("utils.write_safety")',
    'phase._skill_module("utils.write_safety")',
    'load("utils.write_safety")',
])
def test_a_literal_dynamic_import_is_an_edge(tmp_path, call):
    graph = _graph(tmp_path, _in_text_clip(
        f"from importlib import import_module as load\n{call}\n"))
    assert _UP in graph.edges, graph.edges
    assert not graph.computed
    (block,) = upward_problems(graph, known={})
    assert "utils.text_clip (L0 core) imports utils.write_safety (L3 execution)" in block


@pytest.mark.parametrize("statements", [
    'name = "utils.write_safety"\nimportlib.import_module(name)',
    'importlib.import_module(f"utils.{part}")',
    'importlib.import_module(".write_safety", package=package)',
    'importlib.import_module(".write_safety")',
    '__import__(name, fromlist=["redact_text"])',
    '__import__("utils", fromlist=names)',
    'importlib.import_module(*args)',
    'loader = importlib.import_module\nloader("utils.write_safety")',
    'list(map(__import__, ["utils.write_safety"]))',
])
def test_a_computed_dynamic_import_fails_until_its_targets_are_declared(tmp_path, statements):
    graph = _graph(tmp_path, _in_text_clip(statements))
    assert _UP not in graph.edges
    (key,) = graph.computed
    assert key[0] == "utils.text_clip"
    (problem,) = computed_problems(graph, declared={})
    assert problem.startswith("utils.text_clip imports a module whose name is computed")
    assert repr(key[1]) in problem

    declared = {key: (("utils.write_safety",), "the clip reads a configured module")}
    graph = build_graph(tmp_path, declared)
    assert not computed_problems(graph, declared)
    assert _UP in graph.edges
    assert len(upward_problems(graph, known={})) == 1


def test_a_declaration_must_match_a_call_a_reason_and_a_module(tmp_path):
    graph = _graph(tmp_path, {})
    declared = {("utils.text_clip", "importlib.import_module(name)"): (("utils.nothing_*",), " ")}
    assert computed_problems(graph, declared) == [
        "COMPUTED_IMPORTS ('utils.text_clip', 'importlib.import_module(name)'): "
        "the call is gone; delete the entry",
        "COMPUTED_IMPORTS ('utils.text_clip', 'importlib.import_module(name)') "
        "has no reason; give it one",
        "COMPUTED_IMPORTS ('utils.text_clip', 'importlib.import_module(name)') "
        "names 'utils.nothing_*', which matches no module",
    ]


@pytest.mark.parametrize("text", [
    '# importlib.import_module("utils.write_safety")\nimport importlib\n',
    '"""Loads importlib.import_module("utils.write_safety") lazily."""\nimport importlib\n',
    'NOTE = \'__import__("utils.write_safety")\'\n',
    'import importlib\nfrom typing import TYPE_CHECKING\n\n'
    'if TYPE_CHECKING:\n    importlib.import_module("utils.write_safety")\n'
    '    importlib.import_module(computed)\n',
    'import importlib\nimport typing\n\n\ndef clip():\n'
    '    if typing.TYPE_CHECKING:\n        __import__("utils.write_safety")\n',
])
def test_comments_strings_and_type_checking_make_no_edge(tmp_path, text):
    graph = _graph(tmp_path, {"utils/text_clip.py": text})
    assert _UP not in graph.edges
    assert not graph.computed


def test_the_else_branch_of_type_checking_still_counts(tmp_path):
    graph = _graph(tmp_path, {"utils/text_clip.py": (
        "import importlib\nfrom typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n    pass\nelse:\n"
        "    importlib.import_module('utils.write_safety')\n")})
    assert _UP in graph.edges


def test_a_comment_cannot_keep_a_deleted_seed_edge_alive(tmp_path):
    graph = _graph(tmp_path, {"utils/output_handler.py": (
        "import importlib\n\n# Moved here from importlib.import_module(\"utils.plan_updater\").\n"
        "TASKS = 'importlib.import_module(\"utils.plan_updater\")'\n")})
    known = {("utils.output_handler", "utils.plan_updater"): "get_tasks_file_path: test"}
    assert stale_problems(graph, known) == [
        "('utils.output_handler', 'utils.plan_updater'): the import is gone"]


@pytest.mark.parametrize("barrel, providers, imports, target", [
    # An alias of a submodule re-exported through the barrel.
    ("from .providers import result_envelope as envelope\n", "",
     "from utils import envelope\n", "utils.providers.result_envelope"),
    # Two packages, renamed at each hop.
    ("from .providers import Wrapped as Envelope\n",
     "from .result_envelope import ResultEnvelope as Wrapped\n",
     "from utils import Envelope\n", "utils.providers.result_envelope"),
    # An absolute spelling of the same chain.
    ("from utils.providers import Wrapped as Envelope\n",
     "from utils.providers.result_envelope import ResultEnvelope as Wrapped\n",
     "from utils import Envelope\n", "utils.providers.result_envelope"),
])
def test_an_aliased_reexport_resolves_to_the_defining_module(
        tmp_path, barrel, providers, imports, target):
    graph = _graph(tmp_path, {"utils/__init__.py": barrel,
                              "utils/providers/__init__.py": providers,
                              "utils/text_clip.py": imports})
    assert ("utils.text_clip", target) in graph.edges
    assert ("utils.text_clip", "utils.providers") not in graph.edges


@pytest.mark.parametrize("barrel, target", [
    # A binding in the package beats the submodule of the same name.
    ("from .write_safety import redact_text as text_clip\n", "utils.write_safety"),
    ("text_clip = None\n", "utils"),
    ("def text_clip():\n    pass\n", "utils"),
    # With nothing bound, or only the submodule itself, it is the submodule.
    ("", "utils.text_clip"),
    ("from . import text_clip\n", "utils.text_clip"),
    ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
     "    from .write_safety import redact_text as text_clip\n", "utils.text_clip"),
])
def test_a_package_binding_beats_a_submodule(tmp_path, barrel, target):
    graph = _graph(tmp_path, {"utils/__init__.py": barrel,
                              "modes.py": "from utils import text_clip\n"})
    assert [t for (a, t) in graph.edges if a == "modes"] == [target]


@pytest.mark.parametrize("files, statement", [
    ({"modes.py": "from utils import *\n"}, "modes.py:1 from utils import *"),
    ({"utils/__init__.py": "from .write_safety import *\n"},
     "utils/__init__.py:1 from utils.write_safety import *"),
    ({"utils/text_clip.py": "try:\n    from .providers.result_envelope import *\n"
                            "except ImportError:\n    pass\n"},
     "utils/text_clip.py:2 from utils.providers.result_envelope import *"),
])
def test_a_wildcard_import_of_a_skill_module_is_rejected(tmp_path, files, statement):
    graph = _graph(tmp_path, {**files, "utils/output_handler.py": "from os.path import *\n"})
    (problem,) = wildcard_problems(graph)
    assert problem.startswith(statement + ": name the imports instead")
