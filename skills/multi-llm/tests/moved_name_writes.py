"""Find the tests that write a re-exported name on the module that re-exports it.

A module split into owners can keep re-exporting the moved names **for
reading only** (``utils.git_utils``, ``utils.run_status``): a patch of the old
module only shadows the name there and reaches no moved code. Each such
module's moved-names test runs :class:`MovedNameWriteScanner` over every test
file and fails on a write, naming the owner to patch instead.

Every guard scans the whole test tree, and one process often runs several
(``test_layer_moves`` runs one per old module). So
:meth:`MovedNameWriteScanner.scan_dir` parses each file once per version of its
bytes per process, picks out the few nodes a scan can act on (``_candidates``)
and keeps only those, through ``source_cache.derived``. A further guard walks
that short list, not the whole tree, and the process does not hold the test
tree's millions of nodes: every live node slows each garbage collection, and
pytest runs several on the way out.

Every file is still scanned in full. Nothing is skipped on a text search for the
module's name: ``ast.parse`` joins concatenated literals and decodes escapes
(``"utils.git" "_utils.X"``, ``"utils.git\\x5futils.X"``) that such a search
would miss.
"""

import ast
import re

from tests.source_cache import derived, python_files


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _string(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


#: The calls :meth:`MovedNameWriteScanner._call` can report: by the last part of
#: the callee's dotted name, or by its tail. A new write form it learns to read
#: must be admitted here too, or no scan will hand it that call.
_WRITE_CALL_NAMES = frozenset({"patch", "setattr", "delattr"})
_WRITE_CALL_TAILS = ("patch.object", "patch.multiple", "patch.dict")

_ASSIGNMENTS = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)


def _targets(node):
    """The targets of an assignment or ``del`` statement."""
    return node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]


def _leaves(targets):
    """Each target, with tuple, list and starred targets opened up."""
    for target in targets:
        if isinstance(target, (ast.Tuple, ast.List)):
            yield from _leaves(target.elts)
        elif isinstance(target, ast.Starred):
            yield from _leaves([target.value])
        else:
            yield target


def _could_name_a_module(node):
    """Whether *node* has a shape :meth:`MovedNameWriteScanner._is_module` accepts
    for some module: a name, a dotted attribute, ``import_module(...)`` or
    ``sys.modules[...]``. A shape it learns must be admitted here too."""
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _dotted(node) is not None
    if isinstance(node, ast.Call):
        return (_dotted(node.func) or "").endswith("import_module") and bool(node.args)
    if isinstance(node, ast.Subscript):
        return _dotted(node.value) == "sys.modules"
    return False


def _candidates(tree):
    """The nodes of *tree* a scan can act on, in ``ast.walk`` order: imports; an
    assignment or ``del`` with an attribute or subscript target (a write), or an
    assignment whose value could name a module (an alias); and a call whose
    callee could be a write.

    The choice is by shape only, never by module or table, so one list serves
    every scanner and drops nothing any of them would report. The walk order is
    kept: aliases accumulate in it, and hits are reported in it. The list holds
    only these nodes and what is under them, a small part of a test file's tree.
    """
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        elif isinstance(node, _ASSIGNMENTS):
            if any(isinstance(target, (ast.Attribute, ast.Subscript))
                   for target in _leaves(_targets(node))) or (
                    isinstance(node, ast.Assign) and _could_name_a_module(node.value)):
                found.append(node)
        elif isinstance(node, ast.Call):
            func = _dotted(node.func) or ""
            if func.rpartition(".")[2] in _WRITE_CALL_NAMES or func.endswith(_WRITE_CALL_TAILS):
                found.append(node)
    return tuple(found)


class MovedNameWriteScanner:
    """The places a test source writes a moved name on *module*: ``patch``
    and ``patch.object`` / ``patch.multiple`` / ``patch.dict(vars(...))``,
    ``monkeypatch.setattr`` / ``delattr`` (object and dotted-string forms),
    builtin ``setattr`` / ``delattr``, and assignment or ``del`` of the
    attribute. Names reached only through a variable are not seen."""

    def __init__(self, moved, module):
        self.moved = moved
        self.module = module
        #: A patch target string naming a name on the module: ``utils.<module>.X``.
        self.dotted_target = re.compile(rf"(?:^|\.){re.escape(module)}\.([A-Za-z_]\w*)$")

    def scan(self, source, filename="<test>"):
        """The writes in the test source text *source*, parsed here."""
        return self.scan_tree(ast.parse(source, filename))

    def scan_dir(self, root, skip):
        """``(path, hit)`` for every write in a ``*.py`` file under *root*, by path,
        leaving out the file *skip* (the guard's own, which writes on purpose).

        The walk is redone on every call, so a file added or deleted since the last
        one counts. Each file's ``_candidates`` come from ``source_cache.derived``,
        which reads the file again and reuses them only while its bytes hash the
        same, and shares them with every other caller in the process, so they are
        only read here.
        """
        for path in python_files(root):
            if path != skip:
                for hit in self._scan(derived(path, _candidates)):
                    yield path, hit

    def scan_tree(self, tree):
        """The writes in the parsed module *tree*: ``(lineno, form, name, owner)``.

        The tree is walked, never changed, so a tree shared through
        ``source_cache`` stays valid for the next caller.
        """
        return self._scan(_candidates(tree))

    def _scan(self, nodes):
        """The writes among *nodes*, a tree's ``_candidates``."""
        aliases = {self.module}
        for node in nodes:
            if isinstance(node, ast.ImportFrom):
                aliases.update(a.asname or a.name for a in node.names if a.name == self.module)
            elif isinstance(node, ast.Import):
                aliases.update(a.asname for a in node.names if a.asname and a.name.endswith(self.module))
            elif isinstance(node, ast.Assign) and self._is_module(node.value, aliases):
                aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
        hits = []
        for node in nodes:
            if isinstance(node, ast.Call):
                hits.extend(self._call(node, aliases))
            elif isinstance(node, _ASSIGNMENTS):
                form = "del" if isinstance(node, ast.Delete) else "assignment"
                for target in _leaves(_targets(node)):
                    if isinstance(target, ast.Attribute) and self._is_module(target.value, aliases):
                        hits.append((target.lineno, form, target.attr))
                    elif isinstance(target, ast.Subscript) and self._is_namespace(target.value, aliases):
                        hits.append((target.lineno, form, _string(target.slice)))
        return [
            (lineno, form, name, self.moved[name])
            for lineno, form, name in hits if name in self.moved
        ]

    def _is_module(self, node, aliases):
        """Whether *node* is this module: an alias, ``<pkg>.<module>``,
        ``import_module("...<module>")`` or ``sys.modules["...<module>"]``. These
        are the shapes ``_could_name_a_module`` admits; keep the two in step."""
        if isinstance(node, ast.Name):
            return node.id in aliases
        if isinstance(node, ast.Attribute):
            return node.attr == self.module and _dotted(node) is not None
        if isinstance(node, ast.Call) and (_dotted(node.func) or "").endswith("import_module") and node.args:
            return (_string(node.args[0]) or "").split(".")[-1] == self.module
        if isinstance(node, ast.Subscript) and _dotted(node.value) == "sys.modules":
            return (_string(node.slice) or "").split(".")[-1] == self.module
        return False

    def _is_namespace(self, node, aliases):
        """``vars(<module>)`` or ``<module>.__dict__``."""
        if isinstance(node, ast.Call) and _dotted(node.func) == "vars" and node.args:
            return self._is_module(node.args[0], aliases)
        return (isinstance(node, ast.Attribute) and node.attr == "__dict__"
                and self._is_module(node.value, aliases))

    def _call(self, node, aliases):
        """The writes call *node* makes. Only calls ``_candidates`` admits get here
        (``_WRITE_CALL_NAMES``, ``_WRITE_CALL_TAILS``)."""
        func = _dotted(node.func) or ""
        last = func.rpartition(".")[2]
        if last == "patch":
            args = _positional(node, ("target",))
        elif func.endswith("patch.object"):
            args = _positional(node, ("target", "attribute"))
        elif func.endswith("patch.multiple"):
            args = _positional(node, ("target",))
        elif func.endswith("patch.dict"):
            args = _positional(node, ("in_dict", "values"))
        elif last in ("setattr", "delattr"):
            args = _positional(node, ("target", "name"))
        else:
            args = list(node.args)
        if last == "patch" and args and _string(args[0]):
            match = self.dotted_target.search(_string(args[0]))
            if match:
                yield node.lineno, f"{func}(...)", match.group(1)
        elif func.endswith("patch.object") and len(args) >= 2 and self._is_module(args[0], aliases):
            yield node.lineno, f"{func}(...)", _string(args[1])
        elif func.endswith("patch.multiple") and args and (
            self._is_module(args[0], aliases)
            or (_string(args[0]) or "").split(".")[-1] == self.module
        ):
            for keyword in node.keywords:
                if keyword.arg != "target":
                    yield node.lineno, f"{func}(...)", keyword.arg
        elif func.endswith("patch.dict") and args and self._is_namespace(args[0], aliases):
            if len(args) > 1 and isinstance(args[1], ast.Dict):
                for key in args[1].keys:
                    yield node.lineno, f"{func}(...)", _string(key) if key else None
        elif last in ("setattr", "delattr") and args:
            if len(args) >= 2 and self._is_module(args[0], aliases):
                yield node.lineno, f"{func}(...)", _string(args[1])
            elif func != last and _string(args[0]):
                match = self.dotted_target.search(_string(args[0]))
                if match:
                    yield node.lineno, f"{func}(...)", match.group(1)


def _positional(node, names):
    """*node*'s leading arguments as a positional list, filling each position
    from the keyword that names it when it was passed by keyword
    (``mock.patch(target=...)``, ``monkeypatch.setattr(target=..., name=...)``),
    so a keyword spelling is scanned like a positional one."""
    args = list(node.args)
    keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
    for index, name in enumerate(names):
        if index >= len(args):
            if name not in keywords:
                break
            args.append(keywords[name])
    return args
