"""The module-level cache registry (``utils/process_state.py``).

Three parts. The registry itself: :func:`process_cache` hands out a registered
container, and :func:`reset_all` empties every one in place, under the cache's
own lock, or leaves it alone and fails when its owner is mid-update. The
registry is one per process, whichever import route loaded it. And the census
that keeps it whole: every module-level empty container in ``utils/`` is either
registered or listed in :data:`NOT_REGISTERED` with the reason it is not a cache
a test may reset; a registered cache is never rebound, by its module or by a
test, because the registry holds the original object and a rebound name would
leave it clearing a dict nobody reads.

What the census covers. Syntax: every binding at module level, inside
``if``/``try``/``with``/``for`` blocks too, and in the body of a class defined
at module level. A class body runs once per process, so its containers are
process state like a module's; they are named ``<module>.<Class>.<attr>``.
Constructors are recognised through aliases (``D = dict``,
``from collections import OrderedDict as OD``). For what syntax cannot see (a
container a factory returns), two checks run over the modules loaded in a fresh
interpreter: a cache-named container, at module or class level, must be
registered, and every module-level container that is empty right after import
must be registered or explained. Out of scope, by decision: instance
attributes, which die with their object, and memoised functions
(``functools.lru_cache``), which bind no container a module or class holds.
"""

from __future__ import annotations

import ast
import collections
import functools
import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
import types
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from utils import process_state  # noqa: E402

UTILS = SKILL_ROOT / "utils"
TESTS = SKILL_ROOT / "tests"

#: A module-level name made of one of these words is a cache by its name, so an
#: empty container bound to it must be registered; no entry of
#: :data:`NOT_REGISTERED` may excuse one. Words, not substrings: ``_HELD_LOCKS``
#: is not a cache, ``_GATE_WARNED`` is.
CACHE_WORDS = frozenset({
    "CACHE", "CACHES", "MEMO", "MEMOS", "WARNED", "ANNOUNCED", "DECLINED",
    "REPORTED", "DECISIONS", "SEEN",
})

#: Module-level empty containers in ``utils/`` that are not registered, and why.
#: Live bookkeeping (held locks, live children, handles to release at exit) must
#: survive until its owner releases it; clearing it between tests would drop
#: what a test's teardown still has to undo.
NOT_REGISTERED: Dict[str, str] = {
    "utils.codex_trust._RECORDS":
        "armed codex homes that main's finally restores; codex_trust.reset(), "
        "which conftest calls, also clears the registered _REPO_KEYS",
    "utils.git_utils._EMPTY_TREE_IDS":
        "a memo, but git_utils imports nothing from the package while it "
        "initialises (test_git_utils_moved_names); keyed by repository, and "
        "a repository's empty-tree id never changes",
    "utils.improve_workflow._HELD_LOCKS": "phase-dir locks this process holds",
    "utils.landlock_dispatch._SWEPT_BASES":
        "the once-per-process sweep of the real temp directory's slot "
        "leftovers; a test that wants a sweep replaces it, so a reset per "
        "test would only sweep /tmp again",
    "utils.llm_client._LIVE_CHILDREN": "live fan-out children",
    "utils.process_control._JOB_API":
        "the lazily bound Windows job API, fixed for the process",
    "utils.process_control._RUN_JOB_HANDLES": "open handles on the worker's run job",
    "utils.process_control._SIGACTION":
        "the built sigaction call, fixed for the process (kill paths' tests "
        "set it themselves)",
    "utils.process_control._SIGACTION_ABI":
        "the libc ABI, fixed for the process (kill paths' tests set it "
        "themselves)",
    "utils.process_state._ENTRIES": "the registry itself",
    "utils.prompt_files._WINDOWS_FILES":
        "the lazily bound Windows file API, fixed for the process",
    "utils.providers.child_env.ENV_ALLOWLIST_MEASURED":
        "a measurement table, empty on purpose until a parity run fills it",
    "utils.providers.cursor_agent._PROFILE_DIRS":
        "profile directories this process created, removed at exit "
        "(reset_profile_state)",
    "utils.review_orchestrator_base._CLOSING_LINES":
        "a running phase's closing lines, keyed by its directory and popped "
        "at its close",
    "utils.detach._ENTRY_HOLDS": "phase locks a review entry holds until its run ends",
    "utils.detach._PHASE_WORK": "phase-mutating calls still running, joined before the lock is released",
    "utils.process_control._CLI_RECORD":
        "the run whose launch record popen_tree writes in this process (live, "
        "named once by record_cli_identities)",
    "utils.process_control._DETACHED_START":
        "the detached worker's start time, read the moment its pid was known, "
        "for the launcher's failed-handshake teardown",
    "utils.process_control._HELD_PHASE_LOCKS": "phase locks this process holds (re-entrancy)",
    "utils.process_control._LAUNCHES_REFUSED":
        "the run whose worker teardown began: launches stay refused until exit",
    "utils.process_control._PIDFDS":
        "the lazily bound pidfd API, fixed for the process",
    "utils.review_round._BOUND": "the round each running dispatch thread is bound to",
    "utils.state_file._HELD_STATE_LOCKS": "state-file locks this process holds",
    "utils.status_file._HELD_MUTEXES": "interprocess mutexes this process holds",
    "utils.validation._VIEWS_LOCKS":
        "the per-phase host-views mutexes this process holds (re-entrant per thread)",
    "utils.write_safety._HELD_COPY_CALLS":
        "the lazily bound Windows held-copy API, fixed for the process",
}

#: Constructors of an empty container, by the name they are called through.
_EMPTY_CALLS = frozenset({
    "dict", "set", "list", "OrderedDict", "defaultdict", "deque", "Counter",
    "WeakKeyDictionary", "WeakValueDictionary", "WeakSet",
})
#: Of those, the ones whose one argument is a factory, not contents.
_FACTORY_FIRST = frozenset({"defaultdict"})
_REGISTER = "process_cache"
_LOCK_CALLS = frozenset({"Lock", "RLock"})
_KNOWN_CALLS = _EMPTY_CALLS | _LOCK_CALLS | {_REGISTER}

#: What the loaded-module checks count as a container.
_CONTAINER_TYPES = (
    dict, set, list, collections.deque,
    weakref.WeakKeyDictionary, weakref.WeakValueDictionary, weakref.WeakSet,
)


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(SKILL_ROOT).with_suffix("").parts)


def _import_name(path: Path) -> str:
    name = _module_name(path)
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _utils_files() -> Iterator[Path]:
    for path in sorted(UTILS.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def _utils_sources() -> Iterator[Tuple[str, str]]:
    for path in _utils_files():
        yield _module_name(path), path.read_text(encoding="utf-8")


def _words(name: str) -> Set[str]:
    return {word for word in name.upper().split("_") if word}


def _cache_named(path: str) -> bool:
    return bool(_words(path.rsplit(".", 1)[-1]) & CACHE_WORDS)


# ---------------------------------------------------------------------------
# Reading the syntax
# ---------------------------------------------------------------------------


def _call_name(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _Binding(NamedTuple):
    #: The attribute path inside the module: ``_CACHE`` or ``Owner._CACHE``.
    name: str
    #: The expression bound to it, when the statement pairs one with it.
    value: Optional[ast.AST]
    line: int
    how: str


def _target_names(
    target: ast.AST, value: Optional[ast.AST]
) -> Iterator[Tuple[str, Optional[ast.AST]]]:
    """The names an assignment target binds, each paired with its own value
    when the right-hand side lines up with it element by element."""
    if isinstance(target, ast.Name):
        yield target.id, value
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value, None)
    elif isinstance(target, (ast.Tuple, ast.List)):
        pairs = (
            isinstance(value, (ast.Tuple, ast.List))
            and len(value.elts) == len(target.elts)
            and not any(isinstance(e, ast.Starred) for e in [*value.elts, *target.elts])
        )
        values = value.elts if pairs else [None] * len(target.elts)  # type: ignore[union-attr]
        for element, element_value in zip(target.elts, values):
            yield from _target_names(element, element_value)


def _expressions(statement: ast.stmt) -> Iterator[ast.AST]:
    """The expressions a statement evaluates in its own scope (not the
    statements nested in it)."""
    for _, field in ast.iter_fields(statement):
        for item in field if isinstance(field, list) else [field]:
            if isinstance(item, ast.expr):
                yield item
            elif isinstance(item, ast.withitem):
                yield item.context_expr
            elif isinstance(item, ast.keyword):
                yield item.value


def _assignment_expressions(node: ast.AST) -> Iterator[ast.NamedExpr]:
    """``name := value`` in ``node``, outside any nested function scope."""
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.NamedExpr):
            yield current
        stack.extend(
            child for child in ast.iter_child_nodes(current)
            if not isinstance(child, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef))
        )


def _scope_bindings(statements: Iterable[ast.stmt], prefix: str = "") -> Iterator[_Binding]:
    """Every name the statements bind in their own scope: through ``if``,
    ``try``, ``with``, ``for``, ``while`` and ``match`` blocks, and into the
    bodies of classes (as ``Class.name``), never into a function's."""
    for node in statements:
        line = node.lineno
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name, value in _target_names(target, node.value):
                    yield _Binding(prefix + name, value, line, "assignment")
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and isinstance(node.target, ast.Name):
                yield _Binding(prefix + node.target.id, node.value, line, "assignment")
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for name, _ in _target_names(node.target, None):
                yield _Binding(prefix + name, None, line, "for target")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    for name, _ in _target_names(item.optional_vars, None):
                        yield _Binding(prefix + name, None, line, "with target")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    bound = alias.asname or alias.name.split(".")[0]
                    yield _Binding(prefix + bound, None, line, "import")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield _Binding(prefix + node.name, None, line, "def")
        elif isinstance(node, ast.ClassDef):
            yield _Binding(prefix + node.name, None, line, "class")
            yield from _scope_bindings(node.body, f"{prefix}{node.name}.")
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                for name, _ in _target_names(target, None):
                    yield _Binding(prefix + name, None, line, "del")
        for expression in _expressions(node):
            for walrus in _assignment_expressions(expression):
                if isinstance(walrus.target, ast.Name):
                    yield _Binding(
                        prefix + walrus.target.id, walrus.value, walrus.lineno,
                        "assignment expression",
                    )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for handler in getattr(node, "handlers", ()) or ():
            if handler.name:
                yield _Binding(prefix + handler.name, None, handler.lineno, "except target")
            yield from _scope_bindings(handler.body, prefix)
        for case in getattr(node, "cases", ()) or ():
            yield from _scope_bindings(case.body, prefix)
        for field in ("body", "orelse", "finalbody"):
            yield from _scope_bindings(getattr(node, field, ()) or (), prefix)


def _constructor_aliases(tree: ast.Module) -> Dict[str, str]:
    """Each name that calls a known constructor (an empty container, a lock,
    ``process_cache``), mapped to it: the names themselves, import aliases and
    module-level ``D = dict`` assignments, chained."""
    aliases = {name: name for name in _KNOWN_CALLS}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _KNOWN_CALLS:
                    aliases[alias.asname or alias.name] = alias.name
    simple = [
        binding for binding in _scope_bindings(tree.body)
        if "." not in binding.name and isinstance(binding.value, (ast.Name, ast.Attribute))
    ]
    for _ in range(len(simple) + 1):
        grew = False
        for binding in simple:
            value = binding.value
            known = (
                aliases.get(value.id) if isinstance(value, ast.Name)
                else value.attr if value.attr in _KNOWN_CALLS  # type: ignore[union-attr]
                else None
            )
            if known and binding.name not in aliases:
                aliases[binding.name] = known
                grew = True
        if not grew:
            break
    return aliases


def _constructor(node: Optional[ast.AST], aliases: Dict[str, str]) -> Optional[str]:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return aliases.get(func.id)
    if isinstance(func, ast.Attribute):
        return func.attr if func.attr in _KNOWN_CALLS else None
    return None


def _is_empty_container(node: ast.AST, aliases: Dict[str, str]) -> bool:
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, ast.List):
        return not node.elts
    kind = _constructor(node, aliases)
    if kind not in _EMPTY_CALLS:
        return False
    assert isinstance(node, ast.Call)
    return not node.keywords and (
        not node.args or (kind in _FACTORY_FIRST and len(node.args) == 1)
    )


class _Census:
    """What ``utils/`` binds at module and class level, read once per session
    (or, for the guard's own tests, what the given sources bind)."""

    def __init__(self, sources: Optional[Iterable[Tuple[str, str]]] = None) -> None:
        #: ``(module, attribute path) -> (registered name literal or None, line)``
        self.registered: Dict[Tuple[str, str], Tuple[Optional[str], int]] = {}
        #: ``module.attribute path -> line`` of an unregistered empty container
        self.empty: Dict[str, int] = {}
        #: a registered cache bound by anything but ``process_cache``
        self.rebound: List[str] = []
        #: a registered cache used under a module lock it was not registered with
        self.unlocked: List[str] = []
        for module, text in (_utils_sources() if sources is None else sources):
            self._scan(module, ast.parse(text))

    def _scan(self, module: str, tree: ast.Module) -> None:
        aliases = _constructor_aliases(tree)
        bindings = list(_scope_bindings(tree.body))
        lock_of: Dict[str, Optional[str]] = {}
        module_locks: Set[str] = set()
        for binding in bindings:
            kind = _constructor(binding.value, aliases)
            if kind == _REGISTER:
                call = binding.value
                assert isinstance(call, ast.Call)
                keywords = {keyword.arg: keyword.value for keyword in call.keywords}
                first = call.args[0] if call.args else keywords.get("name")
                literal = (
                    first.value
                    if isinstance(first, ast.Constant) and isinstance(first.value, str)
                    else None
                )
                self.registered[(module, binding.name)] = (literal, binding.line)
                lock = keywords.get("lock")
                lock_of[binding.name] = ast.unparse(lock) if lock is not None else None
            elif binding.value is not None and _is_empty_container(binding.value, aliases):
                self.empty[f"{module}.{binding.name}"] = binding.line
            elif kind in _LOCK_CALLS and "." not in binding.name:
                module_locks.add(binding.name)
        for binding in bindings:
            if binding.name in lock_of and _constructor(binding.value, aliases) != _REGISTER:
                self.rebound.append(f"{module}.{binding.name} (line {binding.line}: {binding.how})")
        top = {name for name in lock_of if "." not in name}
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in sorted(top.intersection(node.names)):
                    self.rebound.append(f"{module}.{name} (line {node.lineno}: global)")
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                held = {
                    item.context_expr.id for item in node.items
                    if isinstance(item.context_expr, ast.Name)
                    and item.context_expr.id in module_locks
                }
                if not held:
                    continue
                used = top.intersection(
                    name.id for statement in node.body for name in ast.walk(statement)
                    if isinstance(name, ast.Name)
                )
                for name in sorted(used):
                    if lock_of[name] not in held:
                        self.unlocked.append(
                            f"{module}.{name}: used under `with {', '.join(sorted(held))}` "
                            f"(line {node.lineno}), registered with lock={lock_of[name]}"
                        )

    @property
    def registered_attrs(self) -> Set[str]:
        return {path.rsplit(".", 1)[-1] for (_, path) in self.registered}

    @property
    def registered_dotted(self) -> Set[str]:
        return {f"{mod}.{path}" for (mod, path) in self.registered}


@pytest.fixture(scope="module")
def census() -> _Census:
    return _Census()


#: Calls that replace an attribute: ``monkeypatch.setattr``, ``setattr``,
#: ``patch.object``, ``patch``, ``patch.multiple``.
_REPLACERS = frozenset({"setattr", "object", "patch", "multiple"})
#: The keywords those calls take the attribute, or its dotted path, under.
_NAME_KEYWORDS = frozenset({"name", "attribute", "target"})


def _assigned_attributes(target: ast.AST) -> Iterator[ast.Attribute]:
    if isinstance(target, ast.Attribute):
        yield target
    elif isinstance(target, ast.Starred):
        yield from _assigned_attributes(target.value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _assigned_attributes(element)


def _replacements(tree: ast.AST, where: str, attrs: Set[str], dotted: Set[str]) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name not in _REPLACERS:
                continue
            texts = [
                arg.value for arg in node.args[:2]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ] + [
                keyword.value.value for keyword in node.keywords
                if keyword.arg in _NAME_KEYWORDS
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ]
            for text in texts:
                if text in dotted or (name != "patch" and text in attrs):
                    yield f"{where}:{node.lineno} {name}({text!r}, ...)"
            if name == "multiple":
                for keyword in node.keywords:
                    if keyword.arg in attrs:
                        yield f"{where}:{node.lineno} multiple({keyword.arg}=...)"
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for attribute in _assigned_attributes(target):
                    if attribute.attr in attrs:
                        yield f"{where}:{node.lineno} {ast.unparse(attribute)} = ..."


def _replacements_in(path: Path, attrs: Set[str], dotted: Set[str]) -> Iterator[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    yield from _replacements(tree, str(path.relative_to(SKILL_ROOT)), attrs, dotted)


# ---------------------------------------------------------------------------
# Reading the loaded modules
# ---------------------------------------------------------------------------


def _loaded_containers(module: types.ModuleType) -> Iterator[Tuple[str, Any]]:
    """``(attribute path, object)`` for each container a loaded module binds,
    and each one in the body of a class it defines (Python's own ``_x_`` and
    ``__x__`` class attributes aside)."""
    for attr, value in list(vars(module).items()):
        if attr.startswith("__") and attr.endswith("__"):
            continue
        if isinstance(value, _CONTAINER_TYPES):
            yield attr, value
        elif (
            isinstance(value, type)
            and value.__module__ == module.__name__
            and value.__qualname__ == attr
        ):
            for name, member in list(vars(value).items()):
                if name.startswith("_") and name.endswith("_"):
                    continue
                if isinstance(member, _CONTAINER_TYPES):
                    yield f"{attr}.{name}", member


def _loaded_findings(modules: Iterable[types.ModuleType]) -> Dict[str, List[str]]:
    """What syntax may miss, read from the module objects themselves.

    ``cache_named``: an unregistered container with a cache's name.
    ``empty``: an unregistered module-level container that is empty now (right
    after a fresh import) and not in :data:`NOT_REGISTERED`.
    ``empty_names``: every name of an unregistered empty one, for the stale check.
    """
    seen: Dict[int, Tuple[Any, List[Tuple[str, str]]]] = {}
    for module in modules:
        for path, value in _loaded_containers(module):
            seen.setdefault(id(value), (value, []))[1].append((module.__name__, path))
    cache_named: List[str] = []
    empty: List[str] = []
    empty_names: List[str] = []
    for value, names in seen.values():
        if process_state.is_registered(value):
            continue
        dotted = sorted(f"{module}.{path}" for module, path in names)
        if any(_cache_named(path) for _, path in names):
            cache_named.append(" = ".join(dotted))
        if not value and any("." not in path for _, path in names):
            empty_names.extend(dotted)
            if not set(dotted) & set(NOT_REGISTERED):
                empty.append(" = ".join(dotted))
    return {
        "cache_named": sorted(cache_named),
        "empty": sorted(empty),
        "empty_names": sorted(empty_names),
    }


def _report_loaded() -> None:
    """Import every ``utils`` module and print :func:`_loaded_findings` as JSON
    (run in a fresh interpreter by the ``loaded`` fixture)."""
    modules = [importlib.import_module(_import_name(path)) for path in _utils_files()]
    print(json.dumps(_loaded_findings(modules)))


@pytest.fixture(scope="module")
def loaded(tmp_path_factory) -> Dict[str, List[str]]:
    """The loaded-module findings of a fresh interpreter, where no test has
    filled a cache yet."""
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(SKILL_ROOT)!r})\n"
        f"sys.path.append({str(TESTS)!r})\n"
        "import test_process_state\n"
        "test_process_state._report_loaded()\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
        cwd=tmp_path_factory.mktemp("loaded"),
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(monkeypatch):
    """A private registry, so these tests leave the session's alone."""
    monkeypatch.setattr(process_state, "_ENTRIES", [])
    return process_state


def test_a_cache_is_registered_and_starts_empty(registry):
    cache = registry.process_cache("tests.a._CACHE")
    assert cache == {} and isinstance(cache, dict)
    assert registry.is_registered(cache)
    assert not registry.is_registered({})
    assert registry.registered_names() == ("tests.a._CACHE",)


def test_reset_all_empties_every_kind_in_place(registry):
    as_dict = registry.process_cache("tests.a._CACHE")
    as_set = registry.process_cache("tests.a._WARNED", set)
    as_list = registry.process_cache("tests.a._LINES", list)
    as_dict["k"] = 1
    as_set.add("k")
    as_list.append("k")
    registry.reset_all()
    assert (as_dict, as_set, as_list) == ({}, set(), [])
    assert all(registry.is_registered(c) for c in (as_dict, as_set, as_list))


def test_a_repeated_name_keeps_both_containers(registry):
    """A reloaded module registers its new container beside the old one."""
    old = registry.process_cache("tests.a._CACHE")
    new = registry.process_cache("tests.a._CACHE")
    old["k"] = new["k"] = 1
    registry.reset_all()
    assert old == {} and new == {}
    assert registry.registered_names() == ("tests.a._CACHE", "tests.a._CACHE")


def test_a_cache_is_cleared_under_its_own_lock(registry):
    seen: List[str] = []

    class Recording:
        def acquire(self, timeout: float = -1) -> bool:
            seen.append("acquire")
            return True

        def release(self) -> None:
            seen.append("release")

    class Watched(dict):
        def clear(self) -> None:
            seen.append("clear")
            super().clear()

    registry.process_cache("tests.a._CACHE", Watched, lock=Recording())
    registry.reset_all()
    assert seen == ["acquire", "clear", "release"]


def _hold(lock: Any, name: str) -> Tuple[threading.Thread, threading.Event]:
    """A thread that takes ``lock`` and keeps it until the event is set."""
    holding, release = threading.Event(), threading.Event()

    def owner() -> None:
        with lock:
            holding.set()
            release.wait(10)

    thread = threading.Thread(target=owner, name=name)
    thread.start()
    assert holding.wait(10)
    return thread, release


def test_a_lock_left_held_fails_the_reset_and_names_its_holder(registry, monkeypatch):
    """A thread still holding a cache's lock is mid-update: the reset leaves
    that cache alone, empties the rest, and fails naming the cache and the
    thread, within ``LOCK_WAIT_S``."""
    monkeypatch.setattr(registry, "LOCK_WAIT_S", 0.05)
    lock = threading.Lock()
    held = registry.process_cache("tests.a._CACHE", lock=lock)
    other = registry.process_cache("tests.a._OTHER")
    held["k"] = other["k"] = 1
    thread, release = _hold(lock, "left-running")
    try:
        started = time.monotonic()
        with pytest.raises(registry.CacheBusyError) as failure:
            registry.reset_all()
        assert time.monotonic() - started < 5
    finally:
        release.set()
        thread.join(10)
    assert held == {"k": 1}
    assert other == {}
    message = str(failure.value)
    assert "tests.a._CACHE" in message and "tests.a._OTHER" not in message
    assert "'left-running' is in owner()" in message
    registry.reset_all()
    assert held == {}


def test_a_reset_waits_while_its_owner_is_busy(registry, monkeypatch):
    """``busy`` reads True while a worker computes outside the lock (it holds a
    per-key lock) and will write back: the reset waits it out, or fails."""
    monkeypatch.setattr(registry, "LOCK_WAIT_S", 2.0)
    lock = threading.Lock()
    busy = threading.Event()
    busy.set()
    cache = registry.process_cache("tests.a._CACHE", lock=lock, busy=busy.is_set)
    cache["k"] = 1
    threading.Timer(0.1, busy.clear).start()
    registry.reset_all()
    assert cache == {} and not busy.is_set()

    monkeypatch.setattr(registry, "LOCK_WAIT_S", 0.05)
    busy.set()
    cache["k"] = 1
    with pytest.raises(registry.CacheBusyError, match="tests.a._CACHE"):
        registry.reset_all()
    assert cache == {"k": 1}
    busy.clear()


def test_busy_needs_the_lock_it_is_read_under(registry):
    with pytest.raises(TypeError, match="busy= needs the lock"):
        registry.process_cache("tests.a._CACHE", busy=lambda: False)
    assert registry.registered_names() == ()


def test_a_container_without_clear_is_refused(registry):
    with pytest.raises(TypeError, match="tests.a._TUPLE"):
        registry.process_cache("tests.a._TUPLE", tuple)
    assert registry.registered_names() == ()


def test_nothing_outside_the_tests_resets_the_caches():
    """Caching per process is deliberate (the sandbox probe's once-per-process
    budget, docs/development/status.md): only the suite empties them."""
    callers = []
    for path in sorted(SKILL_ROOT.rglob("*.py")):
        parts = set(path.relative_to(SKILL_ROOT).parts)
        if parts & {"tests", ".venv", "__pycache__"} or path == UTILS / "process_state.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else (
                node.id if isinstance(node, ast.Name) else
                node.name if isinstance(node, ast.alias) else None
            )
            if name == "reset_all":
                callers.append(f"{path.relative_to(SKILL_ROOT)}:{getattr(node, 'lineno', '?')}")
    assert callers == []


# ---------------------------------------------------------------------------
# A reset never mutates a cache its owner is updating (the real modules)
# ---------------------------------------------------------------------------


def _paused_in(
    module: types.ModuleType, needle: str, work: Callable[[], Any], name: str,
) -> Tuple[threading.Thread, threading.Event, List[str]]:
    """Run ``work`` in a thread that stops at the first line of ``module``
    containing ``needle`` until the returned event is set. Returns once the
    thread has stopped there."""
    lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    line = next(number for number, text in enumerate(lines, 1) if needle in text)
    paused, resume = threading.Event(), threading.Event()
    errors: List[str] = []

    def trace(frame, event, arg):
        if (
            event == "line" and frame.f_lineno == line
            and frame.f_code.co_filename == module.__file__ and not paused.is_set()
        ):
            paused.set()
            resume.wait(10)
        return trace

    def body() -> None:
        sys.settrace(trace)
        try:
            work()
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            sys.settrace(None)

    thread = threading.Thread(target=body, name=name)
    thread.start()
    assert paused.wait(10), f"never reached {needle!r}"
    return thread, resume, errors


def test_a_reset_leaves_the_yaml_cache_to_its_eviction_loop(tmp_path, monkeypatch):
    """Paused inside ``load_file``'s eviction loop under ``_LOCK``, between the
    size check and ``next(iter(_CACHE))``: a reset that emptied the dict there
    ended the load with StopIteration."""
    from utils import yaml_io

    monkeypatch.setattr(process_state, "LOCK_WAIT_S", 0.2)
    monkeypatch.setattr(yaml_io, "MAX_ENTRIES", 1)
    monkeypatch.setattr(yaml_io, "is_settled", lambda signature: True)
    document = tmp_path / "doc.yaml"
    document.write_text("answer: 42\n", encoding="utf-8")
    yaml_io._CACHE["older"] = ((), {})
    thread, resume, errors = _paused_in(
        yaml_io, "_CACHE.pop(next(iter(_CACHE)))",
        lambda: yaml_io.load_file(document), "yaml-loader",
    )
    try:
        with pytest.raises(process_state.CacheBusyError) as failure:
            process_state.reset_all()
        assert "older" in yaml_io._CACHE
    finally:
        resume.set()
        thread.join(10)
    assert errors == []
    assert "utils.yaml_io._CACHE: thread 'yaml-loader' is in load_file()" in str(failure.value)
    assert list(yaml_io._CACHE) == [os.path.realpath(document)]


def test_a_reset_leaves_the_shared_ps_table_to_its_reader(monkeypatch):
    """Paused in ``_waiting_process_table`` under ``_SHARED_PS_TABLE_LOCK``,
    between the non-empty check and ``[0]``: a reset that emptied the list
    there ended the read with IndexError."""
    from utils import process_control

    monkeypatch.setattr(process_state, "LOCK_WAIT_S", 0.2)
    monkeypatch.setattr(process_control, "IS_WINDOWS", False)
    monkeypatch.setattr(process_control, "_linux_process_table", lambda: None)
    now = time.monotonic()
    process_control._SHARED_PS_TABLE[:] = [(now, now, {1: "row"})]
    answers: List[Any] = []
    thread, resume, errors = _paused_in(
        process_control, "began, ended, shared = _SHARED_PS_TABLE[0]",
        lambda: answers.append(process_control._waiting_process_table(0, 60)), "ps-reader",
    )
    try:
        with pytest.raises(process_state.CacheBusyError, match="_SHARED_PS_TABLE: thread 'ps-reader'"):
            process_state.reset_all()
    finally:
        resume.set()
        thread.join(10)
    assert errors == []
    assert answers == [{1: "row"}]


def test_a_reset_never_drops_a_held_sandbox_probe_lock(monkeypatch):
    """A probe in flight holds its per-key lock and writes its verdict when
    done. A reset that dropped the lock table there let the next slot on the
    key make a second lock and probe again, and the first probe's verdict
    land in the next test's cache."""
    from utils import sandbox_walk
    from utils import subagent_executor as executor

    monkeypatch.setattr(process_state, "LOCK_WAIT_S", 0.2)
    monkeypatch.setattr(sandbox_walk, "_ladder_has_removed_at", lambda provider, profile: False)
    monkeypatch.setattr(sandbox_walk, "_fallback_rungs", lambda provider, profile: ())
    probing, resume = threading.Event(), threading.Event()
    probes: List[str] = []

    def probe(provider, profile, **kwargs):
        probes.append(threading.current_thread().name)
        probing.set()
        resume.wait(10)
        return executor.SandboxProbeResult("ok", "")

    monkeypatch.setattr(sandbox_walk, "sandbox_verdict", probe)
    answers: List[Any] = []

    def slot() -> None:
        answers.append(executor.cached_sandbox_verdict(
            "fake", object(), "generate", cwd=None, policy=None,
        ))

    first = threading.Thread(target=slot, name="probing-slot")
    second = threading.Thread(target=slot, name="sibling-slot")
    first.start()
    try:
        assert probing.wait(10)
        [(key, held)] = executor._SANDBOX_PROBE_KEY_LOCKS.items()
        with pytest.raises(process_state.CacheBusyError) as failure:
            process_state.reset_all()
        assert executor._SANDBOX_PROBE_KEY_LOCKS == {key: held} and held.locked()
        assert executor._sandbox_probe_lock(key) is held
        second.start()
        deadline = time.monotonic() + 0.3
        while len(probes) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        resume.set()
        first.join(10)
        if second.is_alive() or second.ident is not None:
            second.join(10)
    assert probes == ["probing-slot"]
    assert len(answers) == 2 and answers[0] is answers[1]
    message = str(failure.value)
    assert "utils.probe_state._SANDBOX_PROBE_KEY_LOCKS" in message
    assert "utils.probe_state._SANDBOX_PROBE_CACHE" in message
    process_state.reset_all()
    assert executor._SANDBOX_PROBE_CACHE == {} and executor._SANDBOX_PROBE_KEY_LOCKS == {}


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


def test_every_cache_named_as_one_is_registered(census):
    unregistered = sorted(
        f"{dotted} (utils line {line})"
        for dotted, line in census.empty.items()
        if _cache_named(dotted)
    )
    assert unregistered == [], (
        "a module-level cache must be declared through process_cache so the "
        "suite empties it between tests:\n  NAME = process_cache("
        "\"<module>.NAME\")  # set/list: process_cache(..., set)\n  "
        + "\n  ".join(unregistered)
    )


def test_every_other_empty_container_is_registered_or_explained(census):
    unexplained = sorted(set(census.empty) - set(NOT_REGISTERED))
    assert unexplained == [], (
        "register it with process_cache, or add it to NOT_REGISTERED with the "
        "reason it must outlive a test:\n  " + "\n  ".join(unexplained)
    )


def test_no_entry_of_not_registered_is_stale(census, loaded):
    stale = sorted(set(NOT_REGISTERED) - set(census.empty) - set(loaded["empty_names"]))
    assert stale == [], f"no longer an unregistered empty container: {stale}"
    excused = sorted(d for d in NOT_REGISTERED if _cache_named(d))
    assert excused == [], f"named as a cache, so it must be registered: {excused}"


def test_each_registration_is_named_after_its_binding(census):
    misnamed = sorted(
        f"{module}.{attr}: process_cache({literal!r}) at line {line}"
        for (module, attr), (literal, line) in census.registered.items()
        if literal != f"{module}.{attr}"
    )
    assert misnamed == []


def test_the_registry_holds_the_object_each_module_reads(census):
    """Imported for real: the name each module reads is the registered object."""
    assert len(census.registered) >= 30
    stale = []
    for module, attr in sorted(census.registered):
        value = functools.reduce(getattr, attr.split("."), importlib.import_module(module))
        if not process_state.is_registered(value):
            stale.append(f"{module}.{attr}")
    assert stale == []


def test_no_module_rebinds_a_registered_cache(census):
    assert census.rebound == [], (
        "empty a registered cache in place (.clear()); rebinding the name "
        "leaves the registry clearing the old object:\n  " + "\n  ".join(census.rebound)
    )


def test_a_cache_used_under_its_modules_lock_is_registered_with_it(census):
    """reset_all clears under the lock a cache is registered with; one the
    module guards with a lock it was not given is cleared mid-update."""
    assert census.unlocked == [], "pass lock= to process_cache:\n  " + "\n  ".join(census.unlocked)


def test_no_test_or_module_replaces_a_registered_cache(census):
    """The suite empties caches in place; a test that swaps in a new object
    reads a dict the registry does not hold. Clear it, or fill it, in place:
    ``executor._SANDBOX_PROBE_CACHE.clear()``, ``lines[:] = [...]``, or
    ``patch.dict``."""
    attrs, dotted = census.registered_attrs, census.registered_dotted
    found = [
        hit
        for path in [*sorted(TESTS.rglob("*.py")), *_utils_files()]
        if "__pycache__" not in path.parts
        for hit in _replacements_in(path, attrs, dotted)
    ]
    assert found == []


def test_every_loaded_cache_named_container_is_registered(loaded):
    """Read from the module objects, so a cache syntax cannot see (a
    factory's return value, an alias imported from elsewhere) is held too."""
    assert loaded["cache_named"] == []


def test_every_container_empty_at_import_is_registered_or_explained(loaded):
    assert loaded["empty"] == [], (
        "empty right after import, so a cache whatever its name: register it "
        "or explain it in NOT_REGISTERED:\n  " + "\n  ".join(loaded["empty"])
    )


def test_the_cache_words_match_whole_words_only():
    assert _words("_GATE_WARNED") & CACHE_WORDS
    assert _words("_icon_cache") & CACHE_WORDS
    assert not _words("_HELD_LOCKS") & CACHE_WORDS
    assert not _words("_CACHED_ROWS") & CACHE_WORDS
    assert re.fullmatch(r"[A-Z]+", "".join(sorted(CACHE_WORDS)))


# ---------------------------------------------------------------------------
# What the guard catches (synthetic sources, never utils/)
# ---------------------------------------------------------------------------


def _census_of(source: str) -> _Census:
    return _Census([("utils.example", textwrap.dedent(source))])


@pytest.mark.parametrize("source, dotted", [
    ("D = dict\n_X_CACHE = D()\n", "_X_CACHE"),
    ("from builtins import dict as D\n_NEW_CACHE = D()\n", "_NEW_CACHE"),
    ("from collections import OrderedDict as OD\n_ROWS = OD()\n", "_ROWS"),
    ("import collections\nC = collections.defaultdict\n_ROWS = C(list)\n", "_ROWS"),
    ("D = dict\nE = D\n_ROWS = E()\n", "_ROWS"),
], ids=["assigned", "imported", "imported-ordered", "attribute", "chained"])
def test_the_guard_sees_an_aliased_constructor(source, dotted):
    assert f"utils.example.{dotted}" in _census_of(source).empty


@pytest.mark.parametrize("source, dotted", [
    ("if True:\n    _NEW_CACHE = {}\n", "_NEW_CACHE"),
    ("if X:\n    pass\nelse:\n    _ROWS = []\n", "_ROWS"),
    ("try:\n    import missing\nexcept ImportError:\n    _NEW_CACHE = {}\n", "_NEW_CACHE"),
    ("try:\n    pass\nfinally:\n    _ROWS = set()\n", "_ROWS"),
    ("with open(__file__):\n    _ROWS = dict()\n", "_ROWS"),
    ("for _ in ():\n    pass\nelse:\n    _ROWS = []\n", "_ROWS"),
    ("while False:\n    _ROWS = {}\n", "_ROWS"),
    ("print(_ROWS := {})\n", "_ROWS"),
    ("class Owner:\n    _NEW_CACHE = {}\n", "Owner._NEW_CACHE"),
    ("class A:\n    class B:\n        _ROWS = []\n", "A.B._ROWS"),
    ("if True:\n    class Owner:\n        _ROWS = set()\n", "Owner._ROWS"),
    ("_A_CACHE, _B = {}, None\n", "_A_CACHE"),
    ("[_A, (_B_CACHE, _C)] = [1, ({}, 2)]\n", "_B_CACHE"),
], ids=[
    "if", "else", "except", "finally", "with", "for-else", "while", "walrus",
    "class", "nested-class", "conditional-class", "tuple", "nested-destructuring",
])
def test_the_guard_sees_a_conditional_class_or_unpacked_container(source, dotted):
    assert f"utils.example.{dotted}" in _census_of(source).empty


_REGISTERED = '_CACHE = process_cache("utils.example._CACHE")\n'


@pytest.mark.parametrize("rebinding", [
    "_CACHE, other = {}, None\n",
    "[_CACHE, *rest] = data\n",
    "for _CACHE in ():\n    pass\n",
    "with context() as _CACHE:\n    pass\n",
    "print(_CACHE := {})\n",
    "from elsewhere import _CACHE\n",
    "if X:\n    _CACHE = {}\n",
    "try:\n    pass\nexcept Exception as _CACHE:\n    pass\n",
    "def _CACHE():\n    pass\n",
    "del _CACHE\n",
    "def reset():\n    global _CACHE\n    _CACHE = {}\n",
], ids=[
    "tuple", "starred", "for", "with", "walrus", "import", "conditional",
    "except", "def", "del", "global",
])
def test_the_guard_sees_a_registered_cache_rebound(rebinding):
    census = _census_of(_REGISTERED + rebinding)
    assert census.rebound and all(hit.startswith("utils.example._CACHE ") for hit in census.rebound)


def test_the_guard_accepts_alternative_registrations_and_an_aliased_register():
    census = _census_of(
        "from utils.process_state import process_cache as declare\n"
        "if X:\n"
        '    _CACHE = declare("utils.example._CACHE")\n'
        "else:\n"
        '    _CACHE = declare("utils.example._CACHE", lock=_LOCK)\n'
        "class Owner:\n"
        '    _ROWS = declare("utils.example.Owner._ROWS", list)\n'
    )
    assert census.rebound == [] and census.empty == {}
    assert set(census.registered) == {
        ("utils.example", "_CACHE"), ("utils.example", "Owner._ROWS"),
    }


def test_the_guard_sees_a_cache_registered_without_its_lock():
    census = _census_of(
        "import threading\n"
        "_LOCK = threading.Lock()\n"
        '_CACHE = process_cache("utils.example._CACHE")\n'
        '_GUARDED = process_cache("utils.example._GUARDED", lock=_LOCK)\n'
        "def read(key):\n"
        "    with _LOCK:\n"
        "        return _CACHE.get(key), _GUARDED.get(key)\n"
    )
    assert census.unlocked == [
        "utils.example._CACHE: used under `with _LOCK` (line 6), registered with lock=None"
    ]


_CACHE_ATTRS = ({"_VERSION_CACHE"}, {"utils.subagent_executor._VERSION_CACHE"})


@pytest.mark.parametrize("source", [
    'monkeypatch.setattr(executor, name="_VERSION_CACHE", value={})',
    'monkeypatch.setattr(target=executor, name="_VERSION_CACHE", value={})',
    'monkeypatch.setattr("utils.subagent_executor._VERSION_CACHE", {})',
    'monkeypatch.setattr(target="utils.subagent_executor._VERSION_CACHE", value={})',
    'patch.object(executor, attribute="_VERSION_CACHE", new={})',
    'patch(target="utils.subagent_executor._VERSION_CACHE", new={})',
    'patch.multiple(executor, _VERSION_CACHE={})',
    'executor._VERSION_CACHE, other = {}, 1',
    '[executor._VERSION_CACHE, *rest] = [{}, 1]',
], ids=[
    "setattr-name", "setattr-target-name", "setattr-dotted", "setattr-dotted-keyword",
    "patch-object-attribute", "patch-target", "patch-multiple", "tuple", "starred",
])
def test_the_guard_sees_a_test_replacing_a_cache(source):
    assert list(_replacements(ast.parse(source), "t.py", *_CACHE_ATTRS))


@pytest.mark.parametrize("source", [
    "executor._VERSION_CACHE.clear()",
    'patch.dict(executor._VERSION_CACHE, {"codex": "1.0"})',
    'monkeypatch.setitem(executor._VERSION_CACHE, "codex", "1.0")',
    'monkeypatch.setattr(executor, "_OTHER", {})',
])
def test_the_guard_leaves_in_place_changes_alone(source):
    assert list(_replacements(ast.parse(source), "t.py", *_CACHE_ATTRS)) == []


def test_the_loaded_check_sees_what_syntax_cannot(registry):
    """A cache a factory returns: no literal, no constructor call to read."""
    source = textwrap.dedent('''
        def make():
            return {}
        _NEW_CACHE = make()
        _PROBES = make()
        _RULES = make()
        _RULES["fixed"] = True
        _KEPT_CACHE = process_cache("utils.example._KEPT_CACHE")
        class Owner:
            _SEEN = make()
    ''')
    assert _census_of(source).empty == {}
    module = types.ModuleType("utils.example")
    module.process_cache = registry.process_cache
    exec(compile(source, "<example>", "exec"), module.__dict__)
    findings = _loaded_findings([module])
    assert findings["cache_named"] == ["utils.example.Owner._SEEN", "utils.example._NEW_CACHE"]
    assert findings["empty"] == ["utils.example._NEW_CACHE", "utils.example._PROBES"]
