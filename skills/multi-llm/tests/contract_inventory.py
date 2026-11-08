"""Every key the plugin's machine-readable emitters can write, read off the source.

The inventory half of ``test_result_contract.py`` (code health LLM-6): the schemas
in ``schemas/result_line.schema.json``, ``schemas/status_file.schema.json`` and
``schemas/status_json.schema.json`` describe three outputs a host parses, and a
key an emitter writes but no schema names is the drift those files exist to catch.
A test of real outputs only sees the keys its own scenarios reach, so this module
reads the writers themselves: the ``ResultContext.payload`` builder, every
``write_status``/``update_status``/``update_progress``/``report_progress``/
``claim_run``/``update_run`` call under ``skills/multi-llm``, ``status --json``'s
payload and the run view ``detach.classify_run`` builds.

One walker (:class:`_Walk`) reads every dict-valued expression the same way,
whether it is a writer's argument, one member of it (``progress``) or a local
dict a builder fills in:

* dict literals and their ``**`` spreads; ``dict(...)`` with its positional and
  keyword arguments, nested ones included; ``copy``/``deepcopy``; conditional
  expressions, ``x or {}`` and ``a | b``;
* a module constant used as a key (``RAN_FROM_ROOT_KEY``);
* a local name: every assignment to it, and every mutation of it in the same
  function: ``d[k] = v`` (and ``d[k][j] = v``), ``d.update(...)`` (positional
  and keyword), ``d.setdefault(k, v)`` and ``d |= {...}``, plus the same
  mutations inside a skill function the dict is handed to
  (``self._close_keys(payload)``);
* a call to a function the skill defines: its ``return`` values, read with
  that call's arguments bound to its parameters;
* a parameter: the argument every call of its function in the skill passes
  (``close_ask_run(progress=...)``), or its default when a call leaves it out.

Whatever the walker cannot follow is *unresolved*, never skipped: a key that is
not a constant, a dict read from disk, a call whose result it cannot see, a
parameter whose callers it cannot all find (a ``**kwargs`` collector, a function
passed around as a value or defined twice under one name), a dict handed to a
method it cannot tell apart from another of that name. Each is recorded by
(file, function) with its line and reason, and the test holds them to a named
list, ``PASS_THROUGH``, so a new one is a decision someone makes, not a hole.

The walker does not follow aliases (``other = payload; other["k"] = v``).
"""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple

SKILL_DIR = Path(__file__).resolve().parent.parent

#: Directories under the skill that hold no emitter: the tests themselves and the
#: virtual environment uv makes there.
_SKIPPED_PARTS = {"tests", ".venv", "__pycache__"}

#: The ``utils/status_file.py`` functions that write top-level ``.status.json``
#: keys from their second argument; their calls are found under any alias.
STATUS_WRITERS = ("write_status", "update_status")


def source_files() -> List[Path]:
    """Every Python file under the skill that ships, sorted."""
    return sorted(
        path for path in SKILL_DIR.rglob("*.py")
        if not (_SKIPPED_PARTS & set(path.relative_to(SKILL_DIR).parts))
    )


def _dotted(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _site(module: "_Module", node: ast.AST) -> str:
    return f"{module.rel}:{getattr(node, 'lineno', '?')}"


@dataclass
class Inventory:
    """Keys found, with one example site each, and the sites the walker could
    not follow: (file, function) -> every ``"file:line: reason"`` there."""

    keys: Dict[str, str] = field(default_factory=dict)
    unresolved: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    def add(self, key: str, site: str) -> None:
        self.keys.setdefault(key, site)

    def flag(self, where: Tuple[str, str], why: str) -> None:
        reasons = self.unresolved.setdefault(where, [])
        if why not in reasons:
            reasons.append(why)

    def merge(self, other: "Inventory") -> None:
        for key, site in other.keys.items():
            self.add(key, site)
        for where, reasons in other.unresolved.items():
            for why in reasons:
                self.flag(where, why)


class _Module:
    """One parsed source file, with parent links, its functions by name and the
    names its imports bind."""

    def __init__(self, rel: str, source: str, *, importable: bool = True):
        self.rel = rel
        self.dotted = _dotted(rel)
        self.importable = importable
        self.tree = ast.parse(source, filename=rel)
        self.parents: Dict[ast.AST, ast.AST] = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        self.functions: Dict[str, List[ast.AST]] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(node.name, []).append(node)
        self.imports = self._imports()
        #: Module-level re-exports, ``update_status = _status_file.update_status``.
        self.reexports: Dict[str, ast.AST] = {
            node.targets[0].id: node.value for node in self.tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, (ast.Name, ast.Attribute))
        }
        self._module = None

    @classmethod
    def from_path(cls, path: Path) -> "_Module":
        return cls(path.relative_to(SKILL_DIR).as_posix(), path.read_text(encoding="utf-8"))

    def _imports(self) -> Dict[str, str]:
        """Each name an import binds -> the dotted path it names."""
        package = self.dotted.rsplit(".", 1)[0] if "." in self.dotted else ""
        if self.rel.endswith("__init__.py"):
            package = self.dotted
        bound: Dict[str, str] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        bound[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".", 1)[0]
                        bound[head] = head
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    anchor = package.split(".") if package else []
                    anchor = anchor[: len(anchor) - (node.level - 1)] if node.level > 1 else anchor
                    base = ".".join([*anchor, *([base] if base else [])])
                for alias in node.names:
                    bound[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name
        return bound

    def enclosing_function(self, node: ast.AST) -> Optional[ast.AST]:
        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current
            current = self.parents.get(current)
        return None

    def enclosing_class(self, function: ast.AST) -> Optional[ast.ClassDef]:
        parent = self.parents.get(function)
        return parent if isinstance(parent, ast.ClassDef) else None

    def is_method(self, function: ast.AST) -> bool:
        if self.enclosing_class(function) is None:
            return False
        return not any(isinstance(d, ast.Name) and d.id == "staticmethod"
                       for d in function.decorator_list)  # type: ignore[attr-defined]

    def _top_level_string(self, name: str) -> Optional[str]:
        for node in self.tree.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)):
                return node.value.value
        return None

    def constant(self, node: ast.AST) -> Optional[str]:
        """The string a ``Name`` or ``module.NAME`` key evaluates to, if any."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if not self.importable:
            return self._top_level_string(node.id) if isinstance(node, ast.Name) else None
        if self._module is None:
            self._module = importlib.import_module(self.dotted)
        try:
            if isinstance(node, ast.Name):
                value = getattr(self._module, node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                value = getattr(getattr(self._module, node.value.id), node.attr)
            else:
                return None
        except AttributeError:
            return None
        return value if isinstance(value, str) else None


class _Corpus:
    """Every source module, parsed once, for the walker to look functions and
    their callers up in."""

    def __init__(self, modules: Iterable[_Module]) -> None:
        self.modules = list(modules)
        self.by_rel = {module.rel: module for module in self.modules}
        self.by_dotted = {module.dotted: module for module in self.modules}
        self._index: Optional[Tuple[Dict[str, List[Tuple[_Module, ast.Call]]],
                                    Dict[str, List[Tuple[_Module, ast.AST]]]]] = None

    @classmethod
    def from_sources(cls, sources: Mapping[str, str]) -> "_Corpus":
        """A corpus of made-up files (rel path -> source): their constants come
        from their own top-level ``NAME = "..."`` lines, nothing is imported."""
        return cls(_Module(rel, text, importable=False) for rel, text in sources.items())

    def with_sources(self, sources: Mapping[str, str]) -> "_Corpus":
        """This corpus with some real files' text replaced (constants still come
        from the real modules)."""
        replaced = {rel: _Module(rel, text) for rel, text in sources.items()}
        return _Corpus(replaced.get(m.rel, m) for m in self.modules)

    def definitions(self, name: str) -> List[Tuple[_Module, ast.AST]]:
        return [(m, f) for m in self.modules for f in m.functions.get(name, [])]

    def resolve(self, module: _Module, name: str, depth: int = 0
                ) -> Optional[Tuple[_Module, ast.AST]]:
        """The skill function *name* means in *module*: one it defines, or one it
        imports or re-exports from another skill module. ``None`` for anything
        else (an outside module's function, a class, a builtin)."""
        if depth > 4:
            return None
        if name in module.functions:
            return module, module.functions[name][0]
        value = module.reexports.get(name)
        if isinstance(value, ast.Name):
            return self.resolve(module, value.id, depth + 1) if value.id != name else None
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            owner = self.by_dotted.get(module.imports.get(value.value.id, ""))
            return self.resolve(owner, value.attr, depth + 1) if owner is not None else None
        target = module.imports.get(name)
        if target is not None and "." in target:
            owner = self.by_dotted.get(target.rsplit(".", 1)[0])
            if owner is not None:
                return self.resolve(owner, target.rsplit(".", 1)[-1], depth + 1)
        return None

    def function(self, module: _Module, call: ast.Call) -> Optional[Tuple[_Module, ast.AST]]:
        """The skill function *call* (in *module*) runs, when the walker can tell:
        a name the module defines, imports or re-exports, ``self.name``,
        ``module.name`` for a skill module, or else a name the skill defines once.
        A call into an outside module (``json.load``) or a builtin is ``None``."""
        func = call.func
        if isinstance(func, ast.Name):
            return self.resolve(module, func.id)
        if not isinstance(func, ast.Attribute):
            return None
        name = func.attr
        base = func.value
        if isinstance(base, ast.Name):
            if base.id in ("self", "cls") and name in module.functions:
                return module, module.functions[name][0]
            target = module.imports.get(base.id)
            if target is not None:
                owner = self.by_dotted.get(target)
                return self.resolve(owner, name) if owner is not None else None
        found = self.definitions(name)
        return found[0] if len(found) == 1 else None

    def ambiguous(self, module: _Module, call: ast.Call) -> int:
        """How many skill functions an attribute call on an unknown receiver
        could run, when :meth:`function` cannot pick one (0 otherwise)."""
        func = call.func
        if not isinstance(func, ast.Attribute) or self.function(module, call) is not None:
            return 0
        base = func.value
        if isinstance(base, ast.Name) and (base.id in ("self", "cls") or base.id in module.imports):
            return 0
        count = len(self.definitions(func.attr))
        return count if count > 1 else 0

    def _build_index(self):
        # Another name for a function, anywhere in the skill: an import's
        # `as` name or a module-level re-export. A call under either counts as
        # a call of the original, and the re-export itself is no escape.
        aliases: Dict[str, Set[str]] = {}
        exempt: Set[int] = set()
        for module in self.modules:
            for bound, target in module.imports.items():
                original = target.rsplit(".", 1)[-1]
                if original != bound:
                    aliases.setdefault(bound, set()).add(original)
            for bound, value in module.reexports.items():
                exempt.add(id(value))
                original = value.id if isinstance(value, ast.Name) else value.attr  # type: ignore[union-attr]
                if original != bound:
                    aliases.setdefault(bound, set()).add(original)
        calls: Dict[str, List[Tuple[_Module, ast.Call]]] = {}
        references: Dict[str, List[Tuple[_Module, ast.AST]]] = {}
        for module in self.modules:
            heads: Set[int] = set(exempt)
            for node in ast.walk(module.tree):
                if isinstance(node, ast.Call):
                    heads.add(id(node.func))
                    name = _call_name(node)
                    if name is None:
                        continue
                    for each in {name, *aliases.get(name, ())}:
                        calls.setdefault(each, []).append((module, node))
            for node in ast.walk(module.tree):
                if id(node) in heads or not isinstance(getattr(node, "ctx", None), ast.Load):
                    continue
                if isinstance(node, ast.Name):
                    references.setdefault(node.id, []).append((module, node))
                elif isinstance(node, ast.Attribute):
                    references.setdefault(f".{node.attr}", []).append((module, node))
        self._index = (calls, references)

    def callers(self, module: _Module, function: ast.AST
                ) -> Tuple[List[Tuple[_Module, ast.Call]], List[Tuple[_Module, ast.AST]]]:
        """Every call that may run *function*, and every place it is used as a
        value instead. A bare name counts where it names *function* (the
        module's own, imported or re-exported) and is no local variable; an
        attribute counts anywhere its name matches, unless it is a data
        attribute; a method is reached by attribute only."""
        if self._index is None:
            self._build_index()
        calls, references = self._index  # type: ignore[misc]
        name = function.name  # type: ignore[attr-defined]

        def names_it(m: _Module, node: ast.Name) -> bool:
            if module.is_method(function):
                return False
            enclosing = m.enclosing_function(node)
            if enclosing is not None and _is_local(enclosing, node.id):
                return False
            found = self.resolve(m, node.id)
            return found is not None and found[1] is function

        kept = [(m, c) for m, c in calls.get(name, [])
                if not isinstance(c.func, ast.Name) or names_it(m, c.func)]
        escapes = [(m, n) for m, n in references.get(f".{name}", [])
                   if not _data_attribute(m, n)]
        escapes += [(m, n) for m, n in references.get(name, []) if names_it(m, n)]
        return kept, escapes


def _is_local(function: ast.AST, name: str) -> bool:
    """Whether *name* is a parameter or local variable of *function*. An import
    inside it is not: the module's import table already resolves that name."""
    other = _other_binding(function, name)
    return (_parameter_kind(function, name) is not None
            or next(_local_assignments(function, name), None) is not None
            or (other is not None and not isinstance(other, (ast.Import, ast.ImportFrom))))


def _data_attribute(module: _Module, node: ast.AST) -> bool:
    """Whether ``self.<name>`` at *node* reads a data attribute: the class it
    sits in defines no method of that name (``ResultContext.progress``)."""
    if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self"):
        return False
    current = module.parents.get(node)
    while current is not None and not isinstance(current, ast.ClassDef):
        current = module.parents.get(current)
    if current is None:
        return False
    return not any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and item.name == node.attr for item in current.body)


_CORPUS: Optional[_Corpus] = None


def corpus() -> _Corpus:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = _Corpus(_Module.from_path(path) for path in source_files())
    return _CORPUS


@dataclass(frozen=True)
class _Frame:
    """Where an expression is read: its module and function, and, inside a
    function entered through one call, that call and the caller's frame."""

    module: _Module
    function: Optional[ast.AST]
    call: Optional[ast.Call] = None
    caller: Optional["_Frame"] = None

    @property
    def label(self) -> str:
        return self.function.name if self.function is not None else "<module>"  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _Member:
    key: str
    value: ast.AST
    frame: _Frame
    site: str


_UNKNOWN = object()


def _call_name(node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _local_assignments(function: ast.AST, name: str) -> Iterator[ast.AST]:
    """Each value assigned to the local *name* in *function*."""
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                yield node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name and node.value:
                yield node.value


def _binds(target: ast.AST, name: str) -> bool:
    """Whether the assignment target *target* rebinds *name* itself (a
    subscript or attribute target inside a tuple stores into it instead)."""
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_binds(element, name) for element in target.elts)
    if isinstance(target, ast.Starred):
        return _binds(target.value, name)
    return False


def _other_binding(function: ast.AST, name: str) -> Optional[ast.AST]:
    """A statement that binds *name* in a way the walker does not read: a
    tuple target, a loop or ``with`` variable, ``:=``, ``global``, an import."""
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, (ast.Tuple, ast.List)) and _binds(t, name) for t in node.targets):
                return node
        elif isinstance(node, (ast.For, ast.AsyncFor)) and _binds(node.target, name):
            return node
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            if any(i.optional_vars is not None and _binds(i.optional_vars, name)
                   for i in node.items):
                return node
        elif isinstance(node, ast.NamedExpr) and node.target.id == name:
            return node
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((a.asname or a.name.split(".", 1)[0]) == name for a in node.names):
                return node
    return None


def _assigned_subscripts(node: ast.Assign) -> Iterator[ast.Subscript]:
    """Every ``x[k]`` an assignment stores to, tuple targets included."""
    stack = list(node.targets)
    while stack:
        target = stack.pop()
        if isinstance(target, (ast.Tuple, ast.List)):
            stack.extend(target.elts)
        elif isinstance(target, ast.Subscript):
            yield target


def _path(node: ast.AST, name: str) -> Optional[List[ast.AST]]:
    """The keys from the local *name* down to *node*, when *node* is *name*
    itself (``[]``), ``name[k]...`` or ``name.setdefault(k, ...)...``."""
    keys: List[ast.AST] = []
    while True:
        if isinstance(node, ast.Name):
            return list(reversed(keys)) if node.id == name else None
        if isinstance(node, ast.Subscript):
            keys.append(node.slice)
            node = node.value
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "setdefault" and node.args):
            keys.append(node.args[0])
            node = node.func.value
        else:
            return None


def _dict_node(keys: List[Optional[ast.AST]], values: List[ast.AST], at: ast.AST) -> ast.Dict:
    return ast.copy_location(ast.Dict(keys=keys, values=values), at)


def _parameters(function: ast.AST) -> Tuple[List[ast.arg], List[ast.arg]]:
    args = function.args  # type: ignore[attr-defined]
    return [*args.posonlyargs, *args.args], list(args.kwonlyargs)


def _parameter_kind(function: ast.AST, name: str) -> Optional[str]:
    args = function.args  # type: ignore[attr-defined]
    positional, keyword_only = _parameters(function)
    if any(a.arg == name for a in [*positional, *keyword_only]):
        return "named"
    if args.vararg is not None and args.vararg.arg == name:
        return "*"
    if args.kwarg is not None and args.kwarg.arg == name:
        return "**"
    return None


def _drops_self(module: _Module, function: ast.AST, call: ast.Call) -> bool:
    """Whether *call*'s first positional argument skips *function*'s ``self``:
    a method called through an instance, not through its class."""
    if not module.is_method(function) or not isinstance(call.func, ast.Attribute):
        return False
    owner = module.enclosing_class(function)
    base = call.func.value
    return not (isinstance(base, ast.Name) and owner is not None and base.id == owner.name)


class _Walk:
    """One question put to the source: the keys of an expression, or the
    expressions under one key. *seen* stops a cycle; each question gets its own."""

    def __init__(self, corpus_: _Corpus, inventory: Inventory) -> None:
        self.corpus = corpus_
        self.inventory = inventory
        self.seen: Set[tuple] = set()

    def flag(self, frame: _Frame, node: ast.AST, reason: str) -> None:
        self.inventory.flag((frame.module.rel, frame.label), f"{_site(frame.module, node)}: {reason}")

    def keys(self, node: ast.AST, frame: _Frame) -> None:
        for member in self.members(node, frame):
            self.inventory.add(member.key, member.site)

    # -- one expression ---------------------------------------------------

    def members(self, node: ast.AST, frame: _Frame) -> Iterator[_Member]:
        """Each (key, value) the dict *node* evaluates to can carry."""
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key is None:  # a ** spread
                    yield from self.members(value, frame)
                    continue
                text = frame.module.constant(key)
                if text is None:
                    self.flag(frame, key, f"a key the walker cannot evaluate: {ast.unparse(key)}")
                else:
                    yield _Member(text, value, frame, _site(frame.module, key))
        elif isinstance(node, ast.IfExp):
            yield from self.members(node.body, frame)
            yield from self.members(node.orelse, frame)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                yield from self.members(value, frame)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            yield from self.members(node.left, frame)
            yield from self.members(node.right, frame)
        elif isinstance(node, ast.Constant) and node.value is None:
            return
        elif isinstance(node, ast.Call):
            yield from self._call(node, frame)
        elif isinstance(node, ast.Name):
            yield from self._name(node, frame)
        else:
            self.flag(frame, node, f"a dict the walker cannot follow: {ast.unparse(node)}")

    def _call(self, node: ast.Call, frame: _Frame) -> Iterator[_Member]:
        name = _call_name(node)
        if name == "dict" and isinstance(node.func, ast.Name):
            for arg in node.args:
                yield from self.members(arg, frame)
            for keyword in node.keywords:
                if keyword.arg is None:
                    yield from self.members(keyword.value, frame)
                else:
                    yield _Member(keyword.arg, keyword.value, frame, _site(frame.module, keyword))
            return
        if name in ("copy", "deepcopy"):  # copy.copy(x), copy.deepcopy(x), x.copy()
            if node.args:
                yield from self.members(node.args[0], frame)
                return
            if isinstance(node.func, ast.Attribute):
                yield from self.members(node.func.value, frame)
                return
        marker = ("call", id(node), id(frame.call))
        if marker in self.seen:
            return
        self.seen.add(marker)
        found = self.corpus.function(frame.module, node)
        if found is not None:
            owner, function = found
            returns = [n.value for n in ast.walk(function)
                       if isinstance(n, ast.Return) and n.value is not None]
            if returns:
                callee = _Frame(owner, function, node, frame)
                for value in returns:
                    yield from self.members(value, callee)
                return
        self.flag(frame, node, f"a call whose result the walker cannot read: {ast.unparse(node.func)}(...)")

    # -- one local name ---------------------------------------------------

    def _name(self, node: ast.Name, frame: _Frame) -> Iterator[_Member]:
        name = node.id
        function = frame.function
        if function is None:
            self.flag(frame, node, f"a module-level name: {name}")
            return
        marker = ("name", id(function), name, id(frame.call))
        if marker in self.seen:
            return
        self.seen.add(marker)
        bound = False
        if _parameter_kind(function, name) is not None:
            bound = True
            yield from self._parameter(node, frame)
        for value in _local_assignments(function, name):
            bound = True
            yield from self.members(value, frame)
        other = _other_binding(function, name)
        if other is not None:
            bound = True
            self.flag(frame, other, f"`{name}` is bound by a {type(other).__name__} the walker cannot read")
        if not bound:
            self.flag(frame, node, f"`{name}` has no assignment in {frame.label}")
        yield from self.mutations(name, frame)

    def mutations(self, name: str, frame: _Frame) -> Iterator[_Member]:
        """Every key the function in *frame* adds to its local dict *name* in
        place, and the ones a skill function it hands *name* to adds."""
        assert frame.function is not None
        for node in ast.walk(frame.function):
            if isinstance(node, ast.Assign):
                for target in _assigned_subscripts(node):
                    path = _path(target, name)
                    if path:
                        yield from self._under(path[:-1], _dict_node([path[-1]], [node.value], node), frame)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                path = _path(node.target, name)
                if path:
                    yield from self._under(path[:-1], _dict_node([path[-1]], [node.value], node), frame)
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.BitOr):
                # `d |= {...}`, `d[k] |= {...}`. Any other augmented write
                # (`d[k] += 1`) needs `k` already, so it adds no key; a dict it
                # could grow is a defaultdict or Counter, whose construction
                # the walker reports as a call it cannot read.
                path = _path(node.target, name)
                if path is not None:
                    yield from self._under(path, node.value, frame)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("update", "setdefault"):
                    path = _path(node.func.value, name)
                    if path is not None:
                        yield from self._method_mutation(node, path, frame)
                yield from self._handed_on(node, name, frame)
            elif isinstance(node, ast.Call):
                yield from self._handed_on(node, name, frame)

    def _method_mutation(self, node: ast.Call, path: List[ast.AST], frame: _Frame) -> Iterator[_Member]:
        if node.func.attr == "setdefault":  # type: ignore[attr-defined]
            if not node.args:
                return
            default = node.args[1] if len(node.args) > 1 else ast.copy_location(ast.Constant(None), node)
            yield from self._under(path, _dict_node([node.args[0]], [default], node), frame)
            return
        if len(node.args) > 1:
            self.flag(frame, node, "`.update()` with more than one positional argument")
        keys: List[Optional[ast.AST]] = [None for _ in node.args[:1]]
        values: List[ast.AST] = list(node.args[:1])
        for keyword in node.keywords:
            keys.append(None if keyword.arg is None
                        else ast.copy_location(ast.Constant(keyword.arg), keyword))
            values.append(keyword.value)
        yield from self._under(path, _dict_node(keys, values, node), frame)

    def _under(self, path: List[ast.AST], contents: ast.AST, frame: _Frame) -> Iterator[_Member]:
        """The members of *contents*, as they land under *path* of the dict."""
        for key in reversed(path):
            contents = _dict_node([key], [contents], key)
        yield from self.members(contents, frame)

    def _handed_on(self, call: ast.Call, name: str, frame: _Frame) -> Iterator[_Member]:
        """The mutations a skill function makes to the dict *name* when *call*
        hands it over as an argument."""
        passed = [i for i, arg in enumerate(call.args)
                  if isinstance(arg, ast.Name) and arg.id == name]
        keywords = [k.arg for k in call.keywords
                    if k.arg is not None and isinstance(k.value, ast.Name) and k.value.id == name]
        if not passed and not keywords:
            return
        found = self.corpus.function(frame.module, call)
        if found is None:
            count = self.corpus.ambiguous(frame.module, call)
            if count:
                self.flag(frame, call, f"`{name}` is handed to {ast.unparse(call.func)}(...), "
                          f"which {count} skill functions are named, so what it adds is unknown")
            return  # otherwise a builtin or an outside function: it adds no key of ours
        owner, function = found
        positional, keyword_only = _parameters(function)
        if _drops_self(owner, function, call):
            positional = positional[1:]
        targets = [positional[i].arg for i in passed if i < len(positional)
                   and not any(isinstance(a, ast.Starred) for a in call.args[:i])]
        named = {a.arg for a in [*positional, *keyword_only]}
        targets += [k for k in keywords if k in named]
        callee = _Frame(owner, function, call, frame)
        for parameter in targets:
            marker = ("handed", id(function), parameter, id(call))
            if marker in self.seen:
                continue
            self.seen.add(marker)
            yield from self.mutations(parameter, callee)

    # -- a parameter --------------------------------------------------------

    def _parameter(self, node: ast.Name, frame: _Frame) -> Iterator[_Member]:
        name = node.id
        function = frame.function
        assert function is not None
        kind = _parameter_kind(function, name)
        if kind != "named":
            self.flag(frame, node, f"`{name}` collects the `{kind}` arguments of {frame.label}")
            return
        if frame.call is not None:
            assert frame.caller is not None
            yield from self._argument(frame, name, frame.call, frame.caller)
            return
        definitions = self.corpus.definitions(frame.label)
        if len(definitions) > 1:
            self.flag(frame, node, f"`{name}` is a parameter of {frame.label}, which "
                      f"{len(definitions)} functions are named, so its callers cannot be told apart")
            return
        calls, escapes = self.corpus.callers(frame.module, function)
        if escapes:
            module, where = escapes[0]
            self.flag(frame, node, f"`{name}` is a parameter of {frame.label}, which "
                      f"{_site(module, where)} uses as a value, so not every call is visible")
            return
        if not calls:
            self.flag(frame, node, f"`{name}` is a parameter of {frame.label}, which nothing calls")
            return
        for module, call in calls:
            caller = _Frame(module, module.enclosing_function(call))
            yield from self._argument(frame, name, call, caller)

    def _argument(self, frame: _Frame, name: str, call: ast.Call, caller: _Frame) -> Iterator[_Member]:
        """The members of what *call* passes as parameter *name* of the function in *frame*."""
        function = frame.function
        assert function is not None
        positional, _ = _parameters(function)
        names = [a.arg for a in positional]
        offset = 1 if _drops_self(frame.module, function, call) else 0
        if name in names[offset:]:
            index = names.index(name) - offset
            if (index < len(call.args)
                    and not any(isinstance(a, ast.Starred) for a in call.args[:index + 1])):
                yield from self.members(call.args[index], caller)
                return
        for keyword in call.keywords:
            if keyword.arg == name:
                yield from self.members(keyword.value, caller)
                return
        if (any(isinstance(a, ast.Starred) for a in call.args)
                or any(k.arg is None for k in call.keywords)):
            self.flag(caller, call, f"a `*`/`**` argument to {frame.label} may carry `{name}`")
            return
        default = _default(function, name)
        if default is not None:
            yield from self.members(default, _Frame(frame.module, frame.module.enclosing_function(function)))


def _default(function: ast.AST, name: str) -> Optional[ast.AST]:
    args = function.args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
        if arg.arg == name:
            return default
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if arg.arg == name:
            return default
    return None


# ---------------------------------------------------------------------------
# Questions the inventories ask
# ---------------------------------------------------------------------------

def _argument(node: ast.Call, index: int, keyword: str) -> Optional[ast.AST]:
    if len(node.args) > index:
        return node.args[index]
    for item in node.keywords:
        if item.arg == keyword:
            return item.value
    return None


def _frame_of(module: _Module, node: ast.AST) -> _Frame:
    return _Frame(module, module.enclosing_function(node))


def expression_keys(corpus_: _Corpus, module: _Module, node: ast.AST,
                    frame: Optional[_Frame] = None, inventory: Optional[Inventory] = None
                    ) -> Inventory:
    """Every key the dict-valued expression *node* in *module* can carry."""
    found = inventory if inventory is not None else Inventory()
    _Walk(corpus_, found).keys(node, frame or _frame_of(module, node))
    return found


def member_keys(corpus_: _Corpus, module: _Module, node: ast.AST, name: str,
                inventory: Optional[Inventory] = None) -> Inventory:
    """Every key of the object the dict *node* in *module* can carry under *name*."""
    found = inventory if inventory is not None else Inventory()
    for member in list(_Walk(corpus_, found).members(node, _frame_of(module, node))):
        if member.key == name:
            _Walk(corpus_, found).keys(member.value, member.frame)
    return found


def variable_keys(corpus_: _Corpus, module: _Module, function: ast.AST, name: str,
                  inventory: Optional[Inventory] = None) -> Inventory:
    """Every key the local dict *name* that *function* builds can carry."""
    found = inventory if inventory is not None else Inventory()
    variable = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), function)
    _Walk(corpus_, found).keys(variable, _Frame(module, function))
    return found


def mutation_keys(corpus_: _Corpus, module: _Module, function: ast.AST, name: str,
                  inventory: Optional[Inventory] = None) -> Inventory:
    """The keys *function* adds in place to the dict it calls *name*."""
    found = inventory if inventory is not None else Inventory()
    walk = _Walk(corpus_, found)
    for member in walk.mutations(name, _Frame(module, function)):
        found.add(member.key, member.site)
    return found


def _calls_of(corpus_: _Corpus, rel: str, name: str, inventory: Inventory,
              *, class_name: Optional[str] = None) -> List[Tuple[_Module, ast.Call]]:
    """Every call of the function *name* in *rel*, under whatever name the call
    uses. Where it is used as a value instead, its calls through that value are
    out of sight: that site is unresolved."""
    owner = corpus_.by_rel[rel]
    function = _method(owner, class_name, name) if class_name else owner.functions[name][0]
    calls, escapes = corpus_.callers(owner, function)
    for module, node in escapes:
        inventory.flag((module.rel, _frame_of(module, node).label),
                       f"{_site(module, node)}: {name} used as a value, so its calls "
                       "through that value are out of sight")
    return calls


def _writer_dicts(corpus_: _Corpus, inventory: Inventory) -> Iterator[Tuple[_Module, ast.AST]]:
    """The dict argument of every status writer call."""
    for name in STATUS_WRITERS:
        for module, node in _calls_of(corpus_, "utils/status_file.py", name, inventory):
            value = _argument(node, 1, "updates") or _argument(node, 1, "data")
            if value is not None:
                yield module, value


# ---------------------------------------------------------------------------
# .status.json
# ---------------------------------------------------------------------------

def status_file_top_level(corpus_: Optional[_Corpus] = None) -> Inventory:
    """Top-level ``.status.json`` keys: the writers' second argument, plus
    ``update_run``'s ``extra`` (merged at the top level beside ``run``)."""
    c = corpus_ or corpus()
    inventory = Inventory()
    for module, value in _writer_dicts(c, inventory):
        expression_keys(c, module, value, inventory=inventory)
    for module, node in _calls_of(c, "utils/status_file.py", "update_run", inventory):
        extra = _argument(node, 99, "extra")
        if extra is not None:
            expression_keys(c, module, extra, inventory=inventory)
    # `write_status` stamps this on every write, literal arguments included.
    inventory.add("updated_at", "utils/status_file.py:write_status")
    return inventory


def status_member_keys(name: str, corpus_: Optional[_Corpus] = None) -> Inventory:
    """Keys of the object every ``.status.json`` writer puts under top-level
    *name* (``progress``, ``validation_counts``, ...)."""
    c = corpus_ or corpus()
    inventory = Inventory()
    for module, value in _writer_dicts(c, inventory):
        member_keys(c, module, value, name, inventory)
    return inventory


def progress_keys(corpus_: Optional[_Corpus] = None) -> Inventory:
    """Keys of the reserved ``progress`` object: ``report_progress``'s and
    ``RunHandle.progress``'s keywords, ``update_progress``'s dict and a
    ``progress`` member handed to a status writer."""
    c = corpus_ or corpus()
    inventory = status_member_keys("progress", c)

    def keywords(module: _Module, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg:
                inventory.add(keyword.arg, _site(module, keyword))
            else:
                inventory.flag(
                    (module.rel, _frame_of(module, node).label),
                    f"{_site(module, keyword)}: `**{ast.unparse(keyword.value)}` "
                    f"spread into {_call_name(node)}(...)")

    for module, node in _calls_of(c, "utils/detach.py", "report_progress", inventory):
        keywords(module, node)
    for module, node in _calls_of(c, "utils/detach.py", "progress", inventory,
                                  class_name="RunHandle"):
        if isinstance(node.func, ast.Attribute):
            keywords(module, node)
    for module, node in _calls_of(c, "utils/status_file.py", "update_progress", inventory):
        value = _argument(node, 1, "updates")
        if value is not None:
            expression_keys(c, module, value, inventory=inventory)
    return inventory


def run_keys(corpus_: Optional[_Corpus] = None) -> Inventory:
    """Keys of the reserved ``run`` object: ``claim_run``'s record and every
    ``update_run``; ``append_resolution`` writes ``resolutions`` through the latter."""
    c = corpus_ or corpus()
    inventory = Inventory()
    for module, node in _calls_of(c, "utils/status_file.py", "claim_run", inventory):
        value = _argument(node, 1, "record")
        if value is not None:
            expression_keys(c, module, value, inventory=inventory)
    for module, node in _calls_of(c, "utils/status_file.py", "update_run", inventory):
        value = _argument(node, 1, "updates")
        if value is not None:
            expression_keys(c, module, value, inventory=inventory)
    # `update_run` stamps the id it compared against.
    inventory.add("run_id", "utils/status_file.py:update_run")
    return inventory


def model_entry_keys(corpus_: Optional[_Corpus] = None) -> Inventory:
    """Keys of one ``progress.models`` entry, from the fan-out tracker that owns
    the map (``utils/progress_tracker.py``): its seed literal and every in-place
    write to an ``entry``/``row`` dict there."""
    c = corpus_ or corpus()
    inventory = Inventory()
    module = c.by_rel["utils/progress_tracker.py"]
    for function in module.functions.get("_seed_one", []):
        variable_keys(c, module, function, "entry", inventory)
    for functions in module.functions.values():
        for function in functions:
            for name in ("entry", "row"):
                mutation_keys(c, module, function, name, inventory)
    return inventory


# ---------------------------------------------------------------------------
# The MULTI_LLM_RESULT line
# ---------------------------------------------------------------------------

def _method(module: _Module, class_name: str, method: str) -> ast.AST:
    for node in ast.walk(module.tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return item
    raise LookupError(f"{module.rel}: no {class_name}.{method}")


def result_line_keys(corpus_: Optional[_Corpus] = None) -> Inventory:
    """Every key ``ResultContext.payload`` can put on the line."""
    c = corpus_ or corpus()
    module = c.by_rel["utils/cli_common.py"]
    return variable_keys(c, module, _method(module, "ResultContext", "payload"), "payload")


# ---------------------------------------------------------------------------
# status --json
# ---------------------------------------------------------------------------

#: ``status`` itself and the modules split out of it (``run_status._MOVED``).
STATUS_MODULES = (
    "utils/run_status.py", "utils/status_commands.py", "utils/status_native.py",
    "utils/status_text.py", "utils/status_views.py",
)


def status_json_top_level(corpus_: Optional[_Corpus] = None) -> Inventory:
    """Keys of ``status --json``'s document: ``run_status.main``'s error
    object and the run/all document ``run_status._render_json`` builds."""
    c = corpus_ or corpus()
    module = c.by_rel["utils/run_status.py"]
    inventory = variable_keys(c, module, module.functions["main"][0], "payload")
    return variable_keys(c, module, module.functions["_render_json"][0], "payload", inventory)


def run_view_keys(corpus_: Optional[_Corpus] = None) -> Inventory:
    """Keys of one ``runs[]`` entry: ``detach.classify_run``'s view and every
    in-place write to a ``view`` dict in ``detach`` and the status modules."""
    c = corpus_ or corpus()
    detach = c.by_rel["utils/detach.py"]
    inventory = variable_keys(c, detach, detach.functions["classify_run"][0], "view")
    for rel in STATUS_MODULES:
        status = c.by_rel[rel]
        for functions in status.functions.values():
            for function in functions:
                mutation_keys(c, status, function, "view", inventory)
    return inventory


def workflow_keys(corpus_: Optional[_Corpus] = None) -> Inventory:
    """Top-level keys of the workflow view (``get_workflow_status``'s ``result``)."""
    c = corpus_ or corpus()
    module = c.by_rel["check_workflow_prerequisites.py"]
    return variable_keys(c, module, module.functions["get_workflow_status"][0], "result")


# ---------------------------------------------------------------------------
# Objects built outside the emitters
# ---------------------------------------------------------------------------

def function_keys(rel: str, function: str, variable: Optional[str] = None,
                  *, class_name: Optional[str] = None,
                  corpus_: Optional[_Corpus] = None) -> Inventory:
    """Keys of the dict *function* in *rel* builds: the local *variable* it
    builds up, or, with no variable, the dicts it returns."""
    c = corpus_ or corpus()
    module = c.by_rel[rel]
    node = (_method(module, class_name, function) if class_name
            else module.functions[function][0])
    if variable is not None:
        return variable_keys(c, module, node, variable)
    inventory = Inventory()
    for returned in ast.walk(node):
        if isinstance(returned, ast.Return) and returned.value is not None:
            expression_keys(c, module, returned.value, _Frame(module, node), inventory)
    return inventory


def user_next_step_keys(corpus_: Optional[_Corpus] = None) -> Inventory:
    """``NextPhase.to_dict``'s payload, the note pair ``_close_keys`` adds included."""
    return function_keys("utils/next_phase.py", "to_dict", "payload", class_name="NextPhase",
                         corpus_=corpus_)


def user_resume_keys(corpus_: Optional[_Corpus] = None) -> Inventory:
    return function_keys("utils/next_phase.py", "emit_resume", "resume", corpus_=corpus_)


def commit_range_keys(corpus_: Optional[_Corpus] = None) -> Inventory:
    return function_keys("utils/commit_range.py", "result_payload", "payload", corpus_=corpus_)


def commit_fixes_keys(corpus_: Optional[_Corpus] = None) -> Inventory:
    return function_keys("utils/commit_apply.py", "fix_payload", corpus_=corpus_)
