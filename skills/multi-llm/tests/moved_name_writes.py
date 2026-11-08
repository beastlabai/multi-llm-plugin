"""Find the tests that write a re-exported name on the module that re-exports it.

A module split into owners can keep re-exporting the moved names **for
reading only** (``utils.git_utils``, ``utils.run_status``): a patch of the old
module only shadows the name there and reaches no moved code. Each such
module's moved-names test runs :class:`MovedNameWriteScanner` over every test
file and fails on a write, naming the owner to patch instead.
"""

import ast
import re


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
        tree = ast.parse(source, filename)
        aliases = {self.module}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                aliases.update(a.asname or a.name for a in node.names if a.name == self.module)
            elif isinstance(node, ast.Import):
                aliases.update(a.asname for a in node.names if a.asname and a.name.endswith(self.module))
            elif isinstance(node, ast.Assign) and self._is_module(node.value, aliases):
                aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                hits.extend(self._call(node, aliases))
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
                targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                form = "del" if isinstance(node, ast.Delete) else "assignment"
                for target in self._leaves(targets):
                    if isinstance(target, ast.Attribute) and self._is_module(target.value, aliases):
                        hits.append((target.lineno, form, target.attr))
                    elif isinstance(target, ast.Subscript) and self._is_namespace(target.value, aliases):
                        hits.append((target.lineno, form, _string(target.slice)))
        return [
            (lineno, form, name, self.moved[name])
            for lineno, form, name in hits if name in self.moved
        ]

    def _leaves(self, targets):
        for target in targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                yield from self._leaves(target.elts)
            elif isinstance(target, ast.Starred):
                yield from self._leaves([target.value])
            else:
                yield target

    def _is_module(self, node, aliases):
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
        func = _dotted(node.func) or ""
        last = func.rpartition(".")[2]
        args = node.args
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
