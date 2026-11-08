"""Environment reads in Python source, found by walking the AST.

``test_codex_package.py`` holds the shipped tree's ``MULTI_LLM_*`` reads to
``utils/env_vars.py`` and to the Codex ``.mcp.json`` (D9). This module only
finds the reads: :func:`scan` walks each module, function and class body
(:class:`Scope`), decides which names hold an environment mapping, and
resolves each read's key to the strings it can be, or reports it unresolved.

A read is ``m[k]`` (load), ``m.get/setdefault/pop(k)``, ``k in m`` or
``os.getenv(k)``, where ``m`` is ``os.environ``, a bare ``environ``, a
parameter in :data:`ENV_MAPPING_PARAMS`, or a name bound to any of those or to
a copy of one. A key resolves when it is a string literal; a name bound to one
(a module constant, a local, or a loop variable over a literal or a constant
tuple); an attribute or imported name whose target is a unique module constant
anywhere in the scanned tree; or an f-string or ``+`` of those. Flow-
insensitive on purpose: a name bound twice resolves to both values.

Stdlib only; never imports the code it reads.
"""

from __future__ import annotations

import ast
import dataclasses
import os


def module_constants(tree: ast.Module) -> dict:
    """``{name: value}`` for each top-level assignment of a string literal."""
    found = {}
    for node in tree.body:
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    found[target.id] = value.value
    return found


#: Parameters that hand a function an environment mapping: the
#: ``environ=None`` / ``env=None`` / ``base_env=None`` convention, each falling
#: back to ``os.environ``. A local bound to an environment expression is
#: tracked too (``os.environ if environ is None else environ``,
#: ``dict(os.environ)``, ``os.environ.copy()``, ``{**os.environ, ...}``), and a
#: nested function sees its enclosing function's.
ENV_MAPPING_PARAMS = frozenset({"env", "environ", "base_env"})

#: Our names' prefix. A key that does not resolve but whose static prefix is
#: this (``f"MULTI_LLM_{x}"``, ``"MULTI_LLM_" + x``) is reported apart
#: (:attr:`EnvScan.own_unresolved`): no exemption can excuse a name the
#: inventory cannot list.
OWN_PREFIX = "MULTI_LLM"

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_SCOPE_NODES = _FUNCTION_NODES + (ast.ClassDef,)
_UNKNOWN = ("unknown",)
_ITERATION_WRAPPERS = ("sorted", "tuple", "list", "set", "frozenset", "reversed")


def is_os_environ(node) -> bool:
    return (
        isinstance(node, ast.Attribute) and node.attr == "environ"
        and isinstance(node.value, ast.Name) and node.value.id == "os"
    )


def holds_environment(node, env_names) -> bool:
    """Whether *node* evaluates to an environment mapping: ``os.environ``, a
    bare ``environ``, a tracked name, a copy of one (``.copy()``, ``dict(...)``,
    ``{**...}``), or either branch of a conditional or ``or`` over them."""
    if isinstance(node, ast.IfExp):
        return (holds_environment(node.body, env_names)
                or holds_environment(node.orelse, env_names))
    if isinstance(node, ast.BoolOp):
        return any(holds_environment(value, env_names) for value in node.values)
    if is_os_environ(node):
        return True
    if isinstance(node, ast.Name):
        return node.id == "environ" or node.id in env_names
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "copy" and not node.args:
            return holds_environment(func.value, env_names)
        if isinstance(func, ast.Name) and func.id == "dict" and node.args:
            return holds_environment(node.args[0], env_names)
    if isinstance(node, ast.Dict):
        return any(
            key is None and holds_environment(value, env_names)
            for key, value in zip(node.keys, node.values)
        )
    return False


def _is_getenv(func) -> bool:
    return (
        isinstance(func, ast.Attribute) and func.attr == "getenv"
        and isinstance(func.value, ast.Name) and func.value.id == "os"
    ) or (isinstance(func, ast.Name) and func.id == "getenv")


def _enclosing_parts(node) -> list:
    """The parts of a ``def``, ``lambda`` or ``class`` the enclosing scope evaluates."""
    if isinstance(node, ast.ClassDef):
        return [*node.decorator_list, *node.bases, *node.keywords]
    args = node.args
    parts = [*args.defaults, *[d for d in args.kw_defaults if d is not None]]
    if not isinstance(node, ast.Lambda):
        parts.extend(node.decorator_list)
    return parts


def _own_nodes(scope) -> list:
    """Every node of one module, class or function body, nested scopes excluded."""
    stack = [scope.body] if isinstance(scope, ast.Lambda) else list(scope.body)
    found = []
    while stack:
        node = stack.pop()
        found.append(node)
        if isinstance(node, _SCOPE_NODES):
            stack.extend(_enclosing_parts(node))
        else:
            stack.extend(ast.iter_child_nodes(node))
    return found


def _product(parts) -> frozenset:
    combined = {""}
    for options in parts:
        combined = {head + tail for head in combined for tail in options}
    return frozenset(combined)


class Scope:
    """One module, class or function body: what each name in it is bound to,
    and which names hold an environment mapping. Flow-insensitive: a name bound
    twice resolves to both values, and a parameter to nothing."""

    def __init__(self, node, parent, global_constants: dict):
        self.node, self.parent, self.globals = node, parent, global_constants
        own = "<lambda>" if isinstance(node, ast.Lambda) else getattr(node, "name", "")
        self.qualname = ".".join(n for n in (parent.qualname if parent else "", own) if n)
        self.nodes = _own_nodes(node)
        self.bindings: dict = {}
        params = set()
        if isinstance(node, _FUNCTION_NODES):
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs,
                        *(a for a in (args.vararg, args.kwarg) if a)):
                params.add(arg.arg)
                self._bind_name(arg.arg, _UNKNOWN)
        for item in self.nodes:
            self._bind(item)
        outer = self._visible_parent()
        inherited = outer.env_names - params if outer is not None else set()
        self.env_names = inherited | (params & ENV_MAPPING_PARAMS)
        changed = True
        while changed:
            changed = False
            for name, bindings in self.bindings.items():
                if name not in self.env_names and any(
                    binding[0] == "value" and holds_environment(binding[1], self.env_names)
                    for binding in bindings
                ):
                    self.env_names.add(name)
                    changed = True

    def _visible_parent(self):
        """The nearest enclosing scope whose names this one sees (not a class body)."""
        scope = self.parent
        while scope is not None and isinstance(scope.node, ast.ClassDef):
            scope = scope.parent
        return scope

    def _bind_name(self, name: str, binding) -> None:
        self.bindings.setdefault(name, []).append(binding)

    def _bind_target(self, target, binding) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, binding)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for index, element in enumerate(target.elts):
                if isinstance(element, ast.Name) and binding[0] == "item" and binding[2] is None:
                    self._bind_name(element.id, ("item", binding[1], index))
                else:
                    for name in ast.walk(element):
                        if isinstance(name, ast.Name):
                            self._bind_name(name.id, _UNKNOWN)

    def _bind(self, node) -> None:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                self._bind_target(target, ("value", node.value))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            self._bind_target(node.target, ("value", node.value))
        elif isinstance(node, ast.AugAssign):
            self._bind_target(node.target, _UNKNOWN)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            self._bind_target(node.target, ("item", node.iter, None))
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            self._bind_target(node.optional_vars, _UNKNOWN)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            self._bind_name(node.name, _UNKNOWN)

    def lookup(self, name: str):
        """``(owning scope, bindings)`` for *name*, or None (imported or global)."""
        scope = self
        while scope is not None:
            if name in scope.bindings:
                return scope, scope.bindings[name]
            scope = scope._visible_parent()
        return None

    def _global(self, name: str):
        candidates = self.globals.get(name, ())
        return frozenset(candidates) if len(candidates) == 1 else None

    def values(self, node, seen=frozenset()):
        """Every string *node* can evaluate to, or None when one cannot be known."""
        if isinstance(node, ast.Constant):
            return frozenset({node.value}) if isinstance(node.value, str) else None
        if isinstance(node, ast.Name):
            found = self.lookup(node.id)
            if found is None:
                return self._global(node.id)
            owner, bindings = found
            key = (id(owner), node.id)
            if key in seen:
                return None
            out = set()
            for binding in bindings:
                if binding[0] == "value":
                    got = owner.values(binding[1], seen | {key})
                elif binding[0] == "item":
                    got = owner.items(binding[1], binding[2], seen | {key})
                else:
                    got = None
                if got is None:
                    return None
                out |= got
            return frozenset(out)
        if isinstance(node, ast.Attribute):
            return self._global(node.attr)
        if isinstance(node, ast.JoinedStr):
            parts = [self._fstring_part(part, seen) for part in node.values]
            return None if any(part is None for part in parts) else _product(parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.values(node.left, seen), self.values(node.right, seen)
            return None if left is None or right is None else _product([left, right])
        return None

    def _fstring_part(self, part, seen):
        if isinstance(part, ast.FormattedValue):
            # ``!r`` and ``!a`` quote the value and a format spec reshapes it.
            if part.conversion not in (-1, ord("s")) or part.format_spec is not None:
                return None
            return self.values(part.value, seen)
        return self.values(part, seen)

    def items(self, node, index, seen=frozenset()):
        """The strings iterating *node* yields (element *index* of each, for a
        tuple target), or None: a literal, a name bound to one, or either
        wrapped in ``sorted``/``tuple``/``list``/``set``/``frozenset``/``reversed``."""
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            out = set()
            for element in node.elts:
                if index is not None:
                    if not isinstance(element, (ast.Tuple, ast.List)) or len(element.elts) <= index:
                        return None
                    element = element.elts[index]
                got = self.values(element, seen)
                if got is None:
                    return None
                out |= got
            return frozenset(out)
        if isinstance(node, ast.Name):
            found = self.lookup(node.id)
            if found is None:
                return None
            owner, bindings = found
            key = (id(owner), node.id)
            if key in seen or any(binding[0] != "value" for binding in bindings):
                return None
            out = set()
            for binding in bindings:
                got = owner.items(binding[1], index, seen | {key})
                if got is None:
                    return None
                out |= got
            return frozenset(out)
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _ITERATION_WRAPPERS and len(node.args) == 1
        ):
            return self.items(node.args[0], index, seen)
        return None

    def prefix(self, node, seen=frozenset()) -> str:
        """The part of *node*'s value known before it runs ("" when none is)."""
        known = self.values(node, seen)
        if known is not None:
            return os.path.commonprefix(sorted(known))
        if isinstance(node, ast.JoinedStr):
            head = ""
            for part in node.values:
                got = self._fstring_part(part, seen)
                if got is None or len(got) != 1:
                    if isinstance(part, ast.FormattedValue) and part.conversion in (-1, ord("s")):
                        return head + self.prefix(part.value, seen)
                    return head
                head += next(iter(got))
            return head
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.values(node.left, seen)
            if left is not None and len(left) == 1:
                return next(iter(left)) + self.prefix(node.right, seen)
            return self.prefix(node.left, seen)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return self.prefix(node.left, seen).split("%", 1)[0]
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
        ):
            return self.prefix(node.func.value, seen).split("{", 1)[0]
        if isinstance(node, ast.Name):
            found = self.lookup(node.id)
            if found is None:
                return ""
            owner, bindings = found
            key = (id(owner), node.id)
            if key in seen:
                return ""
            return os.path.commonprefix([
                owner.prefix(binding[1], seen | {key}) if binding[0] == "value" else ""
                for binding in bindings
            ])
        return ""

    def reads(self):
        """``(node, key)`` for every read of an environment mapping in this
        scope: ``m[k]`` (load), ``m.get/setdefault/pop(k)``, ``k in m`` and
        ``os.getenv(k)``, where ``m`` holds an environment mapping."""
        for node in self.nodes:
            if (
                isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                and holds_environment(node.value, self.env_names)
            ):
                yield node, node.slice
            elif isinstance(node, ast.Call) and node.args:
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in ("get", "setdefault", "pop")
                    and holds_environment(func.value, self.env_names)
                ) or _is_getenv(func):
                    yield node, node.args[0]
            elif (
                isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], (ast.In, ast.NotIn))
                and holds_environment(node.comparators[0], self.env_names)
            ):
                yield node, node.left


def scopes(tree: ast.Module, global_constants: dict) -> list:
    found, pending = [], [Scope(tree, None, global_constants)]
    while pending:
        scope = pending.pop()
        found.append(scope)
        pending.extend(
            Scope(node, scope, global_constants)
            for node in scope.nodes if isinstance(node, _SCOPE_NODES)
        )
    return found


def read_keys(tree: ast.Module):
    """Every expression used as a read key in *tree*, in source order."""
    reads = [
        (node.lineno, node.col_offset, key)
        for scope in scopes(tree, {})
        for node, key in scope.reads()
    ]
    return [key for _line, _col, key in sorted(reads, key=lambda r: r[:2])]


@dataclasses.dataclass
class EnvScan:
    """What :func:`scan` found."""

    #: Every name a read resolves to.
    names: set = dataclasses.field(default_factory=set)
    #: ``{path::function: count}`` of keys no rule resolves.
    unresolved: dict = dataclasses.field(default_factory=dict)
    #: ``path:line: key`` for each unresolved key whose static prefix is ours.
    own_unresolved: list = dataclasses.field(default_factory=list)


#: ``{(id(tree), place, constants key): (tree, module's reads)}``. A shared
#: parse (``tests/source_cache``) is scanned once per process; the tree is held
#: so its id cannot be reused by another while the entry lives.
_MODULE_SCANS: dict = {}


def _scan_module(tree, place: str, global_constants: dict, constants_key) -> tuple:
    cache_key = (id(tree), place, constants_key)
    cached = _MODULE_SCANS.get(cache_key)
    if cached is not None and cached[0] is tree:
        return cached[1]
    names, unresolved, own = set(), {}, []
    for scope in scopes(tree, global_constants):
        for node, key in scope.reads():
            values = scope.values(key)
            if values is not None:
                names |= values
            elif scope.prefix(key).startswith(OWN_PREFIX):
                own.append(f"{place}:{node.lineno}: {ast.unparse(key)}")
            else:
                site = f"{place}::{scope.qualname or '<module>'}"
                unresolved[site] = unresolved.get(site, 0) + 1
    result = (frozenset(names), tuple(unresolved.items()), tuple(own))
    _MODULE_SCANS[cache_key] = (tree, result)
    return result


def scan(parsed: dict, root=None) -> EnvScan:
    """Every environment read in *parsed* (``{path: ast.Module}``), each place
    named by its path relative to *root* (as given when *root* is None)."""
    global_constants: dict = {}
    for tree in parsed.values():
        for name, value in module_constants(tree).items():
            global_constants.setdefault(name, set()).add(value)
    constants_key = hash(tuple(sorted(
        (name, tuple(sorted(values))) for name, values in global_constants.items()
    )))

    found = EnvScan()
    for path, tree in parsed.items():
        place = (os.path.relpath(path, root) if root else str(path)).replace(os.sep, "/")
        names, unresolved, own = _scan_module(tree, place, global_constants, constants_key)
        found.names |= names
        for site, count in unresolved:
            found.unresolved[site] = found.unresolved.get(site, 0) + count
        found.own_unresolved.extend(own)
    return found
