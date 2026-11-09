#!/usr/bin/env python3
"""Render ``docs/development/markers.md``: every bracket marker the skill prints.

The orchestrators print lines such as ``[BATCH_DONE]`` or ``[SALVAGE_NEEDED]``;
the host-facing instructions, the references and ``SKILL.md`` tell the driving
agent what to do when it sees one. The two halves are joined only by a string,
so a renamed marker used to break the contract with a green suite. This script
reads both halves with the standard library, writes the join as a table, and
``tests/test_markers_registry.py`` fails when the committed table is stale, when
an instruction cites a marker no source emits, when a new marker is neither
handled by an instruction nor declared not-for-the-host, and when a printed
marker is built in a way no scan can list.

What counts, decided here and nowhere else:

* A **marker** is ``[A-Z][A-Z0-9_]+`` in square brackets: two characters or
  more, since ``[WARN]``, ``[ETA]`` and ``[SKIP]`` are markers the instructions
  act on.
* The **sources** are the skill's non-test ``.py`` files and the hand-written
  ``packaging/`` sources (the codex and opencode servers among them), never the
  generated ``packaging/*/multi-llm/`` copies.
* A marker is **emitted** where a *produced* string *opens a line* with it.
  Produced: the string is printed, written, appended, returned, stored in an
  attribute or a dict value — not tested (``in``, ``==``, ``startswith``,
  ``split``, a regex, a lookup key), which is what a parser does with it. Opens
  a line: at the start of the string or after a newline, behind nothing but
  indentation; a hook message that says "copy it from its ``[NEXT_COMMAND]``
  line" mentions a marker and emits none. Docstrings, comments and bare string
  statements are never produced.
* A string is **resolved** through f-strings, ``+``, ``%`` and ``.format``, and
  through names: module constants (imported ones too, and the names a split
  module re-exports through its ``_MOVED`` table), local assignments and
  loop variables, and a function parameter, from the arguments its callers pass
  (``sweep_live_clis("DEADLINE_SWEEP")`` emits ``[DEADLINE_SWEEP]`` through the
  helper's ``f"[{tag}] …"``). An argument to a function the scan can find is
  decided by the parameter's uses, not by the call. What stays unknown (a model
  name, a dict lookup) is a hole: ``f"[{model}] …"`` prints a label, not a
  marker. The JavaScript sources are read lexically: each string literal is
  produced unless a text test sits right beside it.
* **The construction rule**: a printed bracket that mixes upper-case text with a
  hole (``f"[BATCH_{kind}]"``) is refused, since no scan can list what it
  prints. Write the marker whole, as a literal or a constant.
* A marker is **consumed** where ``SKILL.md``, ``instructions/*.md``,
  ``references/*.md`` or an installable host's shim template
  (``hosts/*/shim.md.tmpl``, to its host what ``SKILL.md`` is to Claude Code)
  cite it with the same brackets. A wildcard such as
  ``[PREREQUISITE_*]`` is not a citation of any one marker.
* The **emitter column names a file and, where the marker's text comes from a
  module constant or a helper's argument, that constant or ``helper()``**
  (``sandbox_rungs.py`` ``SANDBOX_BYPASSED_MARKER``), never a line number. Line numbers
  would make the table stale after any edit above the line; the name is what a
  reader greps for, and an inline literal is found by grepping the marker
  itself.
* The **meaning** is the first sentence of the ``#:`` comment block directly
  above the constant, a trailing comment on its line, or the string statement
  that follows it; ``—`` where none exists. Nothing is invented.

The ``NOT_FOR_HOST`` table holds the host-facing half of the reverse check. Every
emitted marker is treated as host-facing, because the orchestrators print to the
host's terminal and no static rule separates a line the host must act on from a
line it relays. So a marker no instruction cites must be listed here with a
reason key, seeded from the markers that were uncited when the table was made.
An entry whose marker an instruction now cites, or which no file emits any more,
fails the test, so the table can only shrink or be extended on purpose.

Usage (from the repository root)::

    python skills/multi-llm/tests/tools/render_markers.py            # write the doc
    python skills/multi-llm/tests/tools/render_markers.py --check    # exit 1 if stale
    python skills/multi-llm/tests/tools/render_markers.py --stdout   # print, write nothing
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

SKILL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SKILL_ROOT.parent.parent
OUTPUT = REPO_ROOT / "docs" / "development" / "markers.md"

#: The marker shape: an upper-case word of two or more characters in brackets.
MARKER_RE = re.compile(r"\[[A-Z][A-Z0-9_]+\]")

#: Text only known at run time, inside a resolved string.
HOLE = "\x00"

#: A marker that opens a line of a produced string (group 1).
LINE_OPENING_MARKER_RE = re.compile(r"(?:^|\n)[ \t]*(\[[A-Z][A-Z0-9_]+\])")

#: A bracket of upper-case text and at least one hole: the construction rule refuses it.
PARTIAL_MARKER_RE = re.compile(r"\[(?=[A-Z0-9_\x00]*\x00)(?=[A-Z0-9_\x00]*[A-Z])[A-Z0-9_\x00]+\]")

#: Directories the source scan never enters. ``tests`` holds fixtures, not emitters.
SKIP_DIRS = frozenset({"tests", "__pycache__", "node_modules"})

#: ``packaging/<host>/multi-llm/`` are generated copies of the skill, scanned once already.
GENERATED_DIRS = frozenset({"multi-llm"})

#: The hand-written ``packaging/`` sources the scan reads.
PACKAGING_SUFFIXES = (".py", ".mjs", ".js")

#: Where instructions live, relative to the skill root. ``SKILL.md`` is added by name.
CITING_DIRS = ("instructions", "references")

#: The installable hosts' shim templates, relative to the skill root: each is its
#: host's ``SKILL.md``, so a marker only a shim acts on is handled there.
CITING_SHIMS = "hosts/*/shim.md.tmpl"

#: Cells list at most this many emitters / consumers, then ``+N more``.
MAX_EMITTERS = 4
MAX_CONSUMERS = 6

#: More combinations than this and an expression's parts are read one by one.
MAX_TEMPLATES = 64

#: Calls whose result holds their argument's text: ``tuple(MARKERS)``.
_WRAPPERS = frozenset({"tuple", "list", "set", "frozenset", "sorted", "str"})

#: Methods whose result is their receiver's text, reworked: ``f"{MARKER} …".rstrip()``.
_TEXT_METHODS = frozenset({"format", "strip", "lstrip", "rstrip", "expandtabs", "encode", "splitlines"})

#: Methods that test or take apart text: a marker passed to one is being parsed.
_CONSUMER_METHODS = frozenset({
    "startswith", "endswith", "find", "rfind", "index", "rindex", "count", "split",
    "rsplit", "partition", "rpartition", "removeprefix", "removesuffix", "strip",
    "lstrip", "rstrip", "get", "pop", "setdefault", "match", "fullmatch", "search",
    "findall", "finditer",
})

#: Functions (``re.*``, ``len``) that take a marker to find or measure it, not to print it.
_CONSUMER_FUNCTIONS = frozenset({
    "len", "escape", "compile", "search", "match", "fullmatch", "findall", "finditer",
})

REASONS: Dict[str, str] = {
    "operator-cli": (
        "Output of `install-host`, `doctor`, the uv bootstrap or `util instructions`, "
        "read by the person running that command. No workflow phase drives them."
    ),
    "runner-diagnostic": (
        "A sandbox, provider-runner or process-capture diagnostic printed into the "
        "run's output and relayed with it. No instruction branches on it."
    ),
    "metrics-note": (
        "A `utils/metrics.py` warning about the run's timing record. It changes no "
        "decision the host makes."
    ),
    "run-notice": (
        "A one-line notice that an orchestrator adjusted, set aside or refused an input. "
        "The line explains itself; the instruction acts on the exit code and "
        "`MULTI_LLM_RESULT`, not on the marker."
    ),
    "phase-progress": (
        "Progress or outcome text of one orchestrator that the phase's generic "
        "'relay the report' rule already covers."
    ),
    "unnamed-handler": (
        "Printed to the host with a phase report, and reached only through a generic "
        "rule (relay the report, the detached-run text) rather than by name. The "
        "likeliest candidates for a named handler; recorded rather than assumed."
    ),
}

#: marker -> reason key. See the module docstring for how this was decided.
NOT_FOR_HOST: Dict[str, str] = {
    # install-host, doctor, uv bootstrap, util instructions
    "[DRY_RUN]": "operator-cli",
    "[INSTALL_CHECK]": "operator-cli",
    "[INSTALL_CHECK_FAILED]": "operator-cli",
    "[LEGACY_INSTRUCTIONS]": "operator-cli",
    "[NOT_YET_LOADED]": "operator-cli",
    "[SHIM_ALSO_FOUND]": "operator-cli",
    "[SHIM_DIR_REMOVED]": "operator-cli",
    "[SHIM_EDIT_PRESERVED]": "operator-cli",
    "[SHIM_INSTALLED]": "operator-cli",
    "[SHIM_KEPT]": "operator-cli",
    "[SHIM_NOT_FOUND]": "operator-cli",
    "[SHIM_REMOVED]": "operator-cli",
    "[SHIM_SHADOWED]": "operator-cli",
    "[SHIM_UNCHANGED]": "operator-cli",
    "[SHIM_WOULD_REMOVE]": "operator-cli",
    "[UV_MISSING]": "operator-cli",
    # sandbox ladder, provider runners, process capture, abort sweeps
    "[ABORT_SWEEP]": "runner-diagnostic",
    "[DEADLINE_SWEEP]": "runner-diagnostic",
    "[CODEX_HOME_CHECK]": "runner-diagnostic",
    "[INTERRUPT_SWEEP]": "runner-diagnostic",
    "[JAIL_DENIED]": "runner-diagnostic",
    "[MODE_CHECK]": "runner-diagnostic",
    "[PIPE_HELD]": "runner-diagnostic",
    "[PORCELAIN_CHECK]": "runner-diagnostic",
    "[SANDBOX_BYPASSED]": "runner-diagnostic",
    "[SANDBOX_FALLBACK]": "runner-diagnostic",
    "[SANDBOX_INCIDENTAL]": "runner-diagnostic",
    "[SANDBOX_RUNG_CHANGED]": "runner-diagnostic",
    "[SANDBOX_RUNG_RETIRED]": "runner-diagnostic",
    "[SANDBOX_UNAVAILABLE]": "runner-diagnostic",
    "[STREAM_CAP]": "runner-diagnostic",
    "[USAGE_LIMIT]": "runner-diagnostic",
    # metrics
    "[METRICS_CONFLICT]": "metrics-note",
    "[METRICS_DUPLICATE]": "metrics-note",
    "[METRICS_INDEX_MISMATCH]": "metrics-note",
    "[METRICS_NO_TASK_ID]": "metrics-note",
    # an input adjusted, set aside or refused
    "[CHECKPOINT_RECOVERED]": "run-notice",
    "[CONSISTENCY_WARNING]": "run-notice",
    "[CONTEXT_PLAN_SNAPSHOT_CHANGED]": "run-notice",
    "[GIT_CONFIG_CHANGED]": "run-notice",
    "[HEAD_REF_MISMATCH]": "run-notice",
    "[PROMPT_SIDECAR_REWRITTEN]": "run-notice",
    "[PROMPT_SIDECAR_SKIPPED]": "run-notice",
    "[RESULT_NOT_READ]": "run-notice",
    "[RESULT_NOT_SHOWN_NEW]": "run-notice",
    "[RESULT_STALE]": "run-notice",
    "[REVIEW_SCOPE_REGENERATED]": "run-notice",
    "[SALVAGE_RETIRED]": "run-notice",
    "[SALVAGE_WITHHELD]": "run-notice",
    "[SCOPE_BASELINE_MOVED]": "run-notice",
    "[SCOPE_FLAGS_IGNORED]": "run-notice",
    "[SCOPE_POLICY_CHANGED]": "run-notice",
    "[SELECTION_CHANGED]": "run-notice",
    "[STATE_REPLACED]": "run-notice",
    "[VALIDATION_CONTEXT_CITED]": "run-notice",
    "[VALIDATION_CONTEXT_EXCERPTED]": "run-notice",
    "[VALIDATION_CONTEXT_OUTSIDE_ROOT]": "run-notice",
    "[VALIDATION_CONTEXT_REBOUND]": "run-notice",
    "[VALIDATION_CONTEXT_TRUNCATED]": "run-notice",
    "[VALIDATION_RETIRED]": "run-notice",
    "[VALIDATION_TASKS_OVERSIZE]": "run-notice",
    "[WITHHOLD_NOT_RECORDED]": "run-notice",
    # progress and outcome text
    "[ACCEPTED]": "phase-progress",
    "[CONSOLIDATION_SKIPPED]": "phase-progress",
    "[DEADLINE]": "phase-progress",
    "[DRY_RUN_COMPLETE]": "phase-progress",
    "[HIGH_PRIORITY_ISSUES]": "phase-progress",
    "[HUMAN_DECISION_JUDGING_PENDING]": "phase-progress",
    "[REPLY_OK]": "phase-progress",
    "[SKIPPED]": "phase-progress",
    "[WRITE_PHASE_NOT_EXECUTED]": "phase-progress",
    # printed to the host, no named handler
    "[CHECK_FINDINGS]": "unnamed-handler",
    "[DETACHED_RUN]": "unnamed-handler",
    "[NEXT_COMMAND_OPTIONAL]": "unnamed-handler",
    "[NEXT_COMMAND_OPTIONAL_WHY]": "unnamed-handler",
    "[TASK_GENERATION_PENDING]": "unnamed-handler",
}




class Emission(NamedTuple):
    """One place a marker is printed from."""

    path: str            # skill-relative, or ``packaging/...`` repo-relative; POSIX separators
    constant: str        # the module constant, or ``helper()``, that fixes the marker's text; or ""
    meaning: str         # first sentence of the constant's comment/docstring, or ""


class Construction(NamedTuple):
    """A printed bracket built from upper-case text and a run-time value."""

    path: str
    line: int
    text: str            # the resolved string, run-time parts shown as ``{…}``


# ---------------------------------------------------------------------------
# Scanning: files
# ---------------------------------------------------------------------------

def _source_files(skill_root: Path) -> List[Tuple[Path, str]]:
    """``(path, label)`` for every emitting source: the skill, then ``packaging/``."""
    found: List[Tuple[Path, str]] = []
    for path in sorted(skill_root.rglob("*.py")):
        parts = path.relative_to(skill_root).parts[:-1]
        if any(part in SKIP_DIRS or part.startswith(".") for part in parts):
            continue
        found.append((path, path.relative_to(skill_root).as_posix()))
    repo_root = skill_root.parents[1]
    packaging = repo_root / "packaging"
    if packaging.is_dir():
        for path in sorted(packaging.rglob("*")):
            if path.suffix not in PACKAGING_SUFFIXES or not path.is_file():
                continue
            parts = path.relative_to(packaging).parts[:-1]
            if any(part in SKIP_DIRS or part in GENERATED_DIRS or part.startswith(".")
                   for part in parts):
                continue
            found.append((path, path.relative_to(repo_root).as_posix()))
    return found


def _first_sentence(text: str) -> str:
    """The first sentence of a comment block, flattened to one table-safe line."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r":[a-z]+:`~?([^`]*)`", r"`\1`", text)   # :func:`x` -> `x`
    text = text.replace("``", "`")
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z(\[`])", text)
    if match:
        text = text[: match.start()]
    text = text.rstrip(" .;:")
    if len(text) > 140:
        text = text[:140].rsplit(" ", 1)[0].rstrip(" ,;:(") + "…"
    return text.replace("|", "\\|")


def _comment_above(lines: Sequence[str], lineno: int) -> str:
    """The ``#`` comment lines directly above 1-based ``lineno``, blank lines excluded."""
    block: List[str] = []
    index = lineno - 2
    while index >= 0 and lines[index].lstrip().startswith("#"):
        block.append(lines[index].lstrip().lstrip("#").lstrip(":").strip())
        index -= 1
    return " ".join(reversed(block))


def _trailing_comment(line: str) -> str:
    # A "#" inside the marker's own string cannot occur; the first one is the comment.
    if "#" not in line:
        return ""
    after = line.split("#", 1)[1]
    return after.lstrip(":").strip()


def _meaning(lines: Sequence[str], node: ast.stmt, body: Sequence[ast.stmt]) -> str:
    text = _comment_above(lines, node.lineno)
    if not text:
        text = _trailing_comment(lines[node.lineno - 1]) if node.lineno == node.end_lineno else ""
    if not text:
        index = list(body).index(node)
        if index + 1 < len(body):
            follower = body[index + 1]
            value = getattr(follower, "value", None)
            if (isinstance(follower, ast.Expr) and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)):
                text = value.value
    return _first_sentence(text) if text else ""


# ---------------------------------------------------------------------------
# Scanning: what a Python string expression can hold
# ---------------------------------------------------------------------------
#
# A string expression resolves to a few *templates*: its text, with each part
# only known at run time replaced by HOLE. Each piece of text remembers the
# module constant, or the helper call, that fixed it (its *origin*), so the
# registry names what to grep for.

Origin = Optional[Tuple[str, str]]
Template = Tuple[Tuple[str, Origin], ...]
_HOLE_TEMPLATE: Template = ((HOLE, None),)

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _merge(template: Sequence[Tuple[str, Origin]]) -> Template:
    out: List[Tuple[str, Origin]] = []
    for text, origin in template:
        if not text:
            continue
        if out and out[-1][1] == origin:
            out[-1] = (out[-1][0] + text, origin)
        else:
            out.append((text, origin))
    return tuple(out)


def _union(groups: Sequence[Sequence[Template]]) -> List[Template]:
    return list(dict.fromkeys(t for group in groups for t in group))


def _concat(parts: Sequence[Sequence[Template]]) -> List[Template]:
    results: List[Template] = [()]
    for options in parts:
        results = [a + b for a in results for b in options]
        if len(results) > MAX_TEMPLATES:
            # Too many combinations: keep every part on its own between holes,
            # the first one still opening the string, so a marker inside one
            # part is still found where it stands.
            return _union([
                [_merge((() if index == 0 else _HOLE_TEMPLATE) + t + _HOLE_TEMPLATE)
                 for t in options]
                for index, options in enumerate(parts)
            ])
    return list(dict.fromkeys(_merge(t) for t in results))


def _with_origin(templates: Sequence[Template], origin: Origin) -> List[Template]:
    return [_merge([(text, own if own is not None else origin) for text, own in t])
            for t in templates]


#: A ``%`` field: ``%s``, ``%-8d``, ``%(name)s``, and ``%%`` for a literal percent sign.
_PERCENT_FIELD_RE = re.compile(r"%(?:\([^)]*\))?[-#0 +]*\d*(?:\.\d+)?[sdirfxXeEgGc%]")
#: A ``str.format`` field, and the ``{{`` / ``}}`` escapes.
_FORMAT_FIELD_RE = re.compile(r"\{\{|\}\}|\{[^{}]*\}")


def _text_and_origin_at(template: Template) -> Tuple[str, List[Tuple[int, Origin]]]:
    offsets: List[Tuple[int, Origin]] = []
    position = 0
    for text, origin in template:
        offsets.append((position, origin))
        position += len(text)
    return "".join(text for text, _ in template), offsets


def _origin_at(offsets: Sequence[Tuple[int, Origin]], index: int) -> Origin:
    found: Origin = None
    for start, origin in offsets:
        if start > index:
            break
        found = origin
    return found


class _Module:
    """One parsed Python source and the facts the resolver asks of it."""

    def __init__(self, path: Path, label: str, source: str):
        self.label = label
        self.is_package = path.name == "__init__.py"
        stem = label[:-3]
        if self.is_package:
            stem = stem[: -len("/__init__")]
        self.dotted = stem.replace("/", ".")
        self.lines = source.splitlines()
        self.tree = ast.parse(source)
        #: One walk records each node's parent, the function it runs in, the class
        #: around it and the scope whose bindings it belongs to (a function, a
        #: lambda or a class body; ``0`` for the module).
        self.parents: Dict[int, ast.AST] = {}
        self.scopes: Dict[int, Optional[ast.AST]] = {}
        self.owners: Dict[int, Optional[ast.ClassDef]] = {}
        self.members: Dict[int, List[ast.AST]] = {}
        self.nodes: List[ast.AST] = []
        self.docstrings: Set[int] = set()
        imports: List[ast.AST] = []
        stack: List[Tuple[ast.AST, Optional[ast.AST], Optional[ast.ClassDef], int]] = [
            (self.tree, None, None, 0)]
        while stack:
            node, scope, owner, key = stack.pop()
            self.nodes.append(node)
            self.scopes[id(node)] = scope
            self.owners[id(node)] = owner
            self.members.setdefault(key, []).append(node)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                first = node.body[0] if node.body else None
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    self.docstrings.add(id(first.value))
            if isinstance(node, _FUNCTION_NODES):
                inner = (node, owner, id(node))
            elif isinstance(node, ast.ClassDef):
                inner = (scope, node, id(node))
            else:
                inner = (scope, owner, key)
            for child in reversed(list(ast.iter_child_nodes(node))):
                self.parents[id(child)] = node
                stack.append((child,) + inner)
        #: module-level name -> every (statement, value) that assigns it
        self.constants: Dict[str, List[Tuple[ast.stmt, ast.expr]]] = {}
        self.functions: Dict[str, ast.AST] = {}
        self.classes: Dict[str, Dict[str, ast.AST]] = {}
        #: class name -> attribute -> the values its class body assigns
        self.class_constants: Dict[str, Dict[str, List[ast.expr]]] = {}
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.constants.setdefault(target.id, []).append((node, node.value))
            elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                    and node.value is not None):
                self.constants.setdefault(node.target.id, []).append((node, node.value))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = {
                    item.name: item for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                self.class_constants.setdefault(node.name, {}).setdefault(
                                    target.id, []).append(item.value)
        #: name -> the module that defines it, for a split module's read-only
        #: re-exports: a module-level ``_MOVED`` table of sibling owner -> names
        #: (``utils.providers.base``, ``utils.git_utils``, ``utils.run_status``).
        self.reexports: Dict[str, str] = {}
        package = self.dotted if self.is_package else self.dotted.rpartition(".")[0]
        for _statement, value in self.constants.get("_MOVED", ()):
            if not isinstance(value, ast.Dict):
                continue
            for key, names in zip(value.keys, value.values):
                if not (isinstance(key, ast.Constant) and isinstance(names, (ast.Tuple, ast.List))):
                    continue
                for element in names.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        self.reexports[element.value] = (
                            f"{package}.{key.value}" if package else key.value)
        #: local name -> (module, attribute) of ``from module import attribute``
        self.from_imports: Dict[str, Tuple[str, str]] = {}
        #: local name -> module of ``import module [as name]``
        self.module_imports: Dict[str, str] = {}
        package = self.dotted.split(".") if self.is_package else self.dotted.split(".")[:-1]
        for node in imports:
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    keep = len(package) - (node.level - 1)
                    base = ".".join(package[: max(keep, 0)] + ([node.module] if node.module else []))
                for alias in node.names:
                    self.from_imports[alias.asname or alias.name] = (base, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.module_imports[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        self.module_imports[head] = head

    def parent(self, node: ast.AST) -> Optional[ast.AST]:
        return self.parents.get(id(node))

    def scope_of(self, node: ast.AST) -> Optional[ast.AST]:
        """The function or lambda *node* runs in; ``None`` at module level."""
        return self.scopes.get(id(node))

    def class_of(self, node: ast.AST) -> Optional[ast.ClassDef]:
        return self.owners.get(id(node))

    def scope_nodes(self, scope: ast.AST) -> List[ast.AST]:
        """Every node that runs in *scope* itself, nested functions and classes left out
        (their own nodes, not the ``def`` or ``class`` statement)."""
        return [node for node in self.members.get(id(scope), ()) if node is not scope]


def _local_bindings(module: "_Module", scope: ast.AST) -> Dict[str, List[Optional[ast.expr]]]:
    """Name -> the values *scope* binds it to; ``None`` for a value not worth reading."""
    bound: Dict[str, List[Optional[ast.expr]]] = {}

    def add(target: ast.AST, value: Optional[ast.expr]) -> None:
        if isinstance(target, ast.Name):
            bound.setdefault(target.id, []).append(value)
        else:
            for node in ast.walk(target):
                if isinstance(node, ast.Name):
                    bound.setdefault(node.id, []).append(None)

    for node in module.scope_nodes(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                add(target, node.value)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            add(node.target, node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            add(node.target, node.iter if isinstance(node.target, ast.Name) else None)
        elif isinstance(node, ast.NamedExpr):
            add(node.target, node.value)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            add(node.optional_vars, None)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.setdefault(node.name, []).append(None)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not scope:
            bound.setdefault(node.name, []).append(None)
    for node in module.scope_nodes(scope):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bound.pop(name, None)
    return bound


def _parameters(scope: ast.AST) -> Dict[str, Tuple[Optional[int], Optional[ast.expr]]]:
    """Parameter -> (positional index or ``None``, default value or ``None``)."""
    args = scope.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults: List[Optional[ast.expr]] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    params: Dict[str, Tuple[Optional[int], Optional[ast.expr]]] = {}
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        params[arg.arg] = (index, default)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params[arg.arg] = (None, default)
    for arg in (args.vararg, args.kwarg):
        if arg is not None:
            params[arg.arg] = (None, None)
    return params


def _bound_receiver(function: ast.AST, call: ast.Call) -> int:
    """1 when ``obj.method(...)`` fills *function*'s ``self``/``cls`` itself, else 0."""
    positional = function.args.posonlyargs + function.args.args
    return int(bool(positional) and positional[0].arg in ("self", "cls")
               and isinstance(call.func, ast.Attribute))


def _callee_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


class _Scanner:
    """Resolves string expressions and decides which ones print a marker."""

    def __init__(self, modules: Sequence[_Module]):
        self.modules = list(modules)
        self.by_dotted: Dict[str, _Module] = {m.dotted: m for m in self.modules}
        self._memo: Dict[Tuple[str, int, str], List[Template]] = {}
        self._active: Set[Tuple[str, int, str]] = set()
        self._bindings: Dict[int, Dict[str, List[Optional[ast.expr]]]] = {}
        self._callees: Dict[int, Optional[Tuple[_Module, ast.AST]]] = {}
        self._nested_defs: Dict[int, Dict[str, ast.AST]] = {}
        #: id(function) -> the calls that reach it
        self.calls: Dict[int, List[Tuple[_Module, ast.Call]]] = {}
        for module in self.modules:
            for node in module.nodes:
                if isinstance(node, ast.Call):
                    callee = self.callee(module, node)
                    if callee is not None:
                        self.calls.setdefault(id(callee[1]), []).append((module, node))

    # -- names -----------------------------------------------------------

    def defining(self, target: Optional[_Module], attr: str) -> Optional[_Module]:
        """*target*, or the module that defines *attr* when *target* only
        re-exports it (:attr:`_Module.reexports`)."""
        seen: Set[str] = set()
        while (target is not None and attr not in target.constants
               and attr not in target.functions and attr in target.reexports
               and target.dotted not in seen):
            seen.add(target.dotted)
            target = self.by_dotted.get(target.reexports[attr])
        return target

    def module_named(self, module: _Module, local: str) -> Optional[_Module]:
        if local in module.module_imports:
            return self.by_dotted.get(module.module_imports[local])
        if local in module.from_imports:
            base, attr = module.from_imports[local]
            return self.by_dotted.get(f"{base}.{attr}" if base else attr)
        return None

    def _module_of_chain(self, module: _Module, node: ast.expr) -> Optional[_Module]:
        if isinstance(node, ast.Name):
            return self.module_named(module, node.id)
        if isinstance(node, ast.Attribute):
            outer = self._module_of_chain(module, node.value)
            if outer is not None:
                return self.by_dotted.get(f"{outer.dotted}.{node.attr}")
        return None

    def callee(self, module: _Module, call: ast.Call) -> Optional[Tuple[_Module, ast.AST]]:
        """The scanned function a call reaches, when it can be told statically."""
        if id(call) not in self._callees:
            self._callees[id(call)] = self._callee(module, call)
        return self._callees[id(call)]

    def _nested(self, module: _Module, scope: ast.AST) -> Dict[str, ast.AST]:
        if id(scope) not in self._nested_defs:
            self._nested_defs[id(scope)] = {
                node.name: node for node in module.scope_nodes(scope)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
        return self._nested_defs[id(scope)]

    def _callee(self, module: _Module, call: ast.Call) -> Optional[Tuple[_Module, ast.AST]]:
        func = call.func
        if isinstance(func, ast.Name):
            scope = module.scope_of(call)
            while scope is not None:
                if func.id in self._nested(module, scope):
                    return module, self._nested(module, scope)[func.id]
                scope = module.scope_of(scope)
            if func.id in module.functions:
                return module, module.functions[func.id]
            if func.id in module.from_imports:
                base, attr = module.from_imports[func.id]
                target = self.defining(self.by_dotted.get(base), attr)
                if target is not None and attr in target.functions:
                    return target, target.functions[attr]
            return None
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id in ("self", "cls"):
                owner = module.class_of(call)
                if owner is not None:
                    method = module.classes.get(owner.name, {}).get(func.attr)
                    if method is not None:
                        return module, method
                return None
            target = self.defining(self._module_of_chain(module, func.value), func.attr)
            if target is not None and func.attr in target.functions:
                return target, target.functions[func.attr]
        return None

    def resolve_name(self, module: _Module, name: str, scope: Optional[ast.AST]) -> List[Template]:
        key = (module.label, id(scope) if scope is not None else 0, name)
        if key in self._memo:
            return self._memo[key]
        if key in self._active:
            return [_HOLE_TEMPLATE]
        self._active.add(key)
        try:
            result = self._resolve_name(module, name, scope)
        finally:
            self._active.discard(key)
        self._memo[key] = result
        return result

    def _resolve_name(self, module: _Module, name: str, scope: Optional[ast.AST]) -> List[Template]:
        while scope is not None:
            params = _parameters(scope)
            if name in params:
                return self._resolve_parameter(module, scope, name, params[name])
            bindings = self._bindings.get(id(scope))
            if bindings is None:
                bindings = self._bindings[id(scope)] = _local_bindings(module, scope)
            if name in bindings:
                return _union([
                    self.templates(module, value, scope) if value is not None else [_HOLE_TEMPLATE]
                    for value in bindings[name]
                ])
            scope = module.scope_of(scope)
        if name in module.constants:
            return _with_origin(
                _union([self.templates(module, value, None) for _, value in module.constants[name]]),
                (module.label, name),
            )
        if name in module.from_imports:
            base, attr = module.from_imports[name]
            target = self.defining(self.by_dotted.get(base), attr)
            if target is not None and target is not module:
                return self.resolve_name(target, attr, None)
        return [_HOLE_TEMPLATE]

    def _resolve_parameter(self, module: _Module, scope: ast.AST, name: str,
                           param: Tuple[Optional[int], Optional[ast.expr]]) -> List[Template]:
        index, default = param
        if isinstance(scope, ast.Lambda):
            return [_HOLE_TEMPLATE]
        found: List[List[Template]] = []
        for caller, call in self.calls.get(id(scope), ()):
            value: Optional[ast.expr] = None
            for keyword in call.keywords:
                if keyword.arg == name:
                    value = keyword.value
            if value is None and index is not None:
                position = index - _bound_receiver(scope, call)
                if any(isinstance(arg, ast.Starred) for arg in call.args[: position + 1]):
                    found.append([_HOLE_TEMPLATE])
                    continue
                if 0 <= position < len(call.args):
                    value = call.args[position]
            if value is not None:
                found.append(_with_origin(
                    self.templates(caller, value, caller.scope_of(call)),
                    (caller.label, f"{scope.name}()"),
                ))
            elif default is not None:
                found.append(self.templates(module, default, module.scope_of(scope)))
        if not found and default is not None:
            found.append(self.templates(module, default, module.scope_of(scope)))
        return _union(found) or [_HOLE_TEMPLATE]

    # -- expressions -----------------------------------------------------

    def templates(self, module: _Module, node: ast.expr, scope: Optional[ast.AST]) -> List[Template]:
        """What *node* can hold; never empty (an unknown is one hole)."""
        return self._templates(module, node, scope) or [_HOLE_TEMPLATE]

    def _templates(self, module: _Module, node: ast.expr, scope: Optional[ast.AST]) -> List[Template]:
        if isinstance(node, ast.Constant):
            return [((node.value, None),)] if isinstance(node.value, str) else [_HOLE_TEMPLATE]
        if isinstance(node, ast.JoinedStr):
            parts: List[List[Template]] = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append([((str(value.value), None),)])
                elif (isinstance(value, ast.FormattedValue) and value.conversion == -1
                        and value.format_spec is None):
                    parts.append(self.templates(module, value.value, scope))
                else:
                    parts.append([_HOLE_TEMPLATE])
            return _concat(parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _concat([self.templates(module, node.left, scope),
                            self.templates(module, node.right, scope)])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            values = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
            return self._fill(module, scope, self.templates(module, node.left, scope),
                              _PERCENT_FIELD_RE, list(values), {})
        if isinstance(node, ast.IfExp):
            return _union([self.templates(module, node.body, scope),
                           self.templates(module, node.orelse, scope)])
        if isinstance(node, ast.BoolOp):
            return _union([self.templates(module, value, scope) for value in node.values])
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return _union([self.templates(module, element, scope) for element in node.elts])
        if isinstance(node, ast.Starred):
            return self.templates(module, node.value, scope)
        if isinstance(node, ast.Name):
            return self.resolve_name(module, node.id, scope)
        if isinstance(node, ast.Attribute):
            target = self.defining(self._module_of_chain(module, node.value), node.attr)
            if target is not None:
                return self.resolve_name(target, node.attr, None)
            owner = module.class_of(node)
            if (owner is not None and isinstance(node.value, ast.Name)
                    and node.value.id in ("self", "cls", owner.name)):
                values = module.class_constants.get(owner.name, {}).get(node.attr, [])
                return _union([self.templates(module, value, None) for value in values])
            return [_HOLE_TEMPLATE]
        if isinstance(node, ast.Subscript):
            return self.templates(module, node.value, scope)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "format":
                return self._fill(module, scope, self.templates(module, func.value, scope),
                                  _FORMAT_FIELD_RE, list(node.args),
                                  {k.arg: k.value for k in node.keywords if k.arg})
            if isinstance(func, ast.Attribute) and func.attr in _TEXT_METHODS:
                return self.templates(module, func.value, scope)
            if isinstance(func, ast.Name) and func.id in _WRAPPERS and len(node.args) == 1:
                return self.templates(module, node.args[0], scope)
        return [_HOLE_TEMPLATE]

    def _fill(self, module: _Module, scope: Optional[ast.AST], templates: Sequence[Template],
              pattern: "re.Pattern[str]", positional: List[ast.expr],
              named: Dict[str, ast.expr]) -> List[Template]:
        """*templates* with each ``%`` or ``.format`` field replaced by what fills it."""
        out: List[List[Template]] = []
        for template in templates:
            pieces: List[List[Template]] = []
            counter = 0
            for text, origin in template:
                last = 0
                for match in pattern.finditer(text):
                    pieces.append([((text[last:match.start()], origin),)])
                    last = match.end()
                    field = match.group(0)
                    if field in ("%%", "{{", "}}"):
                        pieces.append([((field[0], origin),)])
                        continue
                    value: Optional[ast.expr] = None
                    inner = field[1:-1] if field.startswith("{") else ""
                    if field.startswith("%("):
                        value = None
                    elif field.startswith("%") or inner == "":
                        value = positional[counter] if counter < len(positional) else None
                        counter += 1
                    elif inner.isdigit():
                        index = int(inner)
                        value = positional[index] if index < len(positional) else None
                    elif inner.isidentifier():
                        value = named.get(inner)
                    plain = field in ("%s", "{}") or inner.isdigit() or inner.isidentifier()
                    if value is not None and plain and not isinstance(value, ast.Starred):
                        pieces.append(self.templates(module, value, scope))
                    else:
                        pieces.append([_HOLE_TEMPLATE])
                pieces.append([((text[last:], origin),)])
            out.append(_concat(pieces))
        return _union(out)

    # -- where an expression's value goes --------------------------------

    def role(self, module: _Module, node: ast.AST) -> str:
        """``producer``, ``consumer`` or ``neutral``: what the code does with *node*'s text.

        A consumer tests text for the marker (``in``, ``==``, ``startswith``,
        ``split``, a regex, a lookup key); a producer passes it on (``print``,
        ``append``, ``return``, a write, a dict value, an attribute). An
        assignment to a plain name, a loop's iterable and an argument to a
        scanned function are neutral: the name's, the loop variable's or the
        parameter's own uses decide, since the resolver reads through them.
        """
        current = node
        while True:
            parent = module.parent(current)
            if isinstance(parent, ast.IfExp) and current is parent.test:
                return "neutral"
            if isinstance(parent, (ast.Tuple, ast.List, ast.Set, ast.Starred, ast.IfExp, ast.BoolOp)):
                current = parent
                continue
            if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Mod) and current is parent.left:
                current = parent
                continue
            if isinstance(parent, ast.Subscript):
                if current is parent.value:
                    current = parent
                    continue
                return "consumer"
            keyword = None
            if isinstance(parent, ast.keyword):
                keyword, current, parent = parent, parent, module.parent(parent)
            if isinstance(parent, ast.Call):
                if current is parent.func:
                    return "neutral"
                name = _callee_name(parent.func)
                if isinstance(parent.func, ast.Name) and name in _WRAPPERS:
                    current = parent
                    continue
                first = bool(parent.args) and current is parent.args[0]
                if isinstance(parent.func, ast.Attribute):
                    if name in _CONSUMER_METHODS:
                        return "consumer"
                    if name in ("replace", "sub", "subn") and first:
                        return "consumer"
                if name in _CONSUMER_FUNCTIONS:
                    return "consumer"
                callee = self.callee(module, parent)
                if callee is not None and self._reaches_a_parameter(callee[1], parent, current, keyword):
                    return "neutral"
                return "producer"
            if isinstance(parent, (ast.Compare, ast.Assert, ast.If, ast.While)):
                return "consumer" if isinstance(parent, (ast.Compare, ast.Assert)) else "neutral"
            if isinstance(parent, ast.Dict):
                return "consumer" if current in parent.keys else "producer"
            if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                return "neutral" if all(isinstance(t, ast.Name) for t in targets) else "producer"
            if isinstance(parent, (ast.NamedExpr, ast.For, ast.AsyncFor, ast.comprehension,
                                   ast.arguments, ast.Expr, ast.withitem)):
                return "neutral"
            return "producer"

    @staticmethod
    def _reaches_a_parameter(function: ast.AST, call: ast.Call, argument: ast.AST,
                             keyword: Optional[ast.keyword]) -> bool:
        """Does *argument* land in a named parameter of *function*, whose uses decide?"""
        if isinstance(function, ast.Lambda):
            return False
        args = function.args
        if keyword is not None:
            return keyword.arg in {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        position = next((i for i, arg in enumerate(call.args) if arg is argument), None)
        if position is None or any(isinstance(arg, ast.Starred) for arg in call.args[: position + 1]):
            return False
        return position + _bound_receiver(function, call) < len(args.posonlyargs + args.args)

    # -- the scan --------------------------------------------------------

    def sites(self, module: _Module) -> List[ast.AST]:
        """Each outermost string expression, and each name that may hold one."""
        found: List[ast.AST] = []
        for node in module.nodes:
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, str) or "[" not in node.value:
                    continue
                if id(node) in module.docstrings:
                    continue
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
                pass
            elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
                pass
            elif isinstance(node, ast.JoinedStr):
                pass
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _TEXT_METHODS):
                pass
            else:
                continue
            parent = module.parent(node)
            if isinstance(parent, (ast.JoinedStr, ast.FormattedValue)):
                continue
            if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add):
                continue
            if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Mod) and node is parent.left:
                continue
            if isinstance(parent, ast.Attribute) and node is parent.value:
                continue
            if isinstance(parent, ast.Call) and node is parent.func:
                continue
            found.append(node)
        return found

    def scan(self) -> Tuple[Dict[str, List[Emission]], List[Construction]]:
        seen: Dict[str, Set[Tuple[str, str]]] = {}
        constructions: Set[Construction] = set()
        for module in self.modules:
            for node in self.sites(module):
                if self.role(module, node) != "producer":
                    continue
                for template in self.templates(module, node, module.scope_of(node)):
                    text, offsets = _text_and_origin_at(template)
                    if "[" not in text:
                        continue
                    for match in LINE_OPENING_MARKER_RE.finditer(text):
                        origin = _origin_at(offsets, match.start(1) + 1)
                        where = origin if origin is not None else (module.label, "")
                        seen.setdefault(match.group(1), set()).add(where)
                    for match in PARTIAL_MARKER_RE.finditer(text):
                        constructions.add(Construction(
                            module.label, node.lineno, match.group(0).replace(HOLE, "{…}")))
        by_label = {m.label: m for m in self.modules}
        emitted: Dict[str, List[Emission]] = {}
        for marker, places in seen.items():
            named = {path for path, constant in places if constant}
            rows = []
            for path, constant in sorted(places):
                if not constant and path in named:
                    continue      # the file's constant row says where; a bare sighting adds nothing
                rows.append(Emission(path, constant, self._meaning_of(by_label, path, constant)))
            emitted[marker] = rows
        return emitted, sorted(constructions)

    @staticmethod
    def _meaning_of(by_label: Dict[str, _Module], path: str, constant: str) -> str:
        module = by_label.get(path)
        if module is None or not constant or constant.endswith("()"):
            return ""
        statement = module.constants[constant][0][0]
        return _meaning(module.lines, statement, module.tree.body)


# ---------------------------------------------------------------------------
# Scanning: the opencode server and other JavaScript sources
# ---------------------------------------------------------------------------
#
# Lexical, not parsed: string literals outside comments, a template literal's
# ``${…}`` read as a run-time part. A literal right after a text test
# (``.startsWith(``, ``===``, ``new RegExp(``) or right before one is a consumer;
# any other is a producer. The opencode server reads the markers it parses from
# the shipped surface, so none of its own literals is a parser today.

_JS_CONSUMER_BEFORE = re.compile(
    r"(?:\.(?:startsWith|endsWith|includes|indexOf|lastIndexOf|split|replace|replaceAll"
    r"|match|matchAll|search|has|get)\(|new RegExp\(|[!=]==?|\bcase)\s*$")
_JS_CONSUMER_AFTER = re.compile(r"^\s*(?:[!=]==?|\)\s*\.test\()")
_JS_REGEX_BEFORE = re.compile(r"(?:[(,=:\[!&|?{};]|\breturn|\btypeof)\s*$")
_JS_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}


def _js_strings(source: str) -> List[Tuple[int, int, str]]:
    """``(start, end, text)`` of each string literal outside comments and regexes."""
    found: List[Tuple[int, int, str]] = []
    index, size = 0, len(source)
    while index < size:
        char = source[index]
        if source.startswith("//", index):
            newline = source.find("\n", index)
            index = size if newline < 0 else newline
            continue
        if source.startswith("/*", index):
            close = source.find("*/", index + 2)
            index = size if close < 0 else close + 2
            continue
        if char == "/" and _JS_REGEX_BEFORE.search(source[max(0, index - 20):index]):
            cursor, in_class = index + 1, False
            while cursor < size and source[cursor] != "\n":
                if source[cursor] == "\\":
                    cursor += 2
                    continue
                if source[cursor] == "[":
                    in_class = True
                elif source[cursor] == "]":
                    in_class = False
                elif source[cursor] == "/" and not in_class:
                    break
                cursor += 1
            index = cursor + 1
            continue
        if char in "'\"`":
            cursor, text = index + 1, []
            while cursor < size and source[cursor] != char:
                if source[cursor] == "\\" and cursor + 1 < size:
                    text.append(_JS_ESCAPES.get(source[cursor + 1], source[cursor + 1]))
                    cursor += 2
                    continue
                if char == "`" and source.startswith("${", cursor):
                    depth, cursor = 1, cursor + 2
                    while cursor < size and depth:
                        depth += {"{": 1, "}": -1}.get(source[cursor], 0)
                        cursor += 1
                    text.append(HOLE)
                    continue
                if char != "`" and source[cursor] == "\n":
                    break
                text.append(source[cursor])
                cursor += 1
            found.append((index, cursor + 1, "".join(text)))
            index = cursor + 1
            continue
        index += 1
    return found


def _scan_js(label: str, source: str) -> Tuple[Dict[str, Set[Tuple[str, str]]], Set[Construction]]:
    seen: Dict[str, Set[Tuple[str, str]]] = {}
    constructions: Set[Construction] = set()
    for start, end, text in _js_strings(source):
        if "[" not in text:
            continue
        if (_JS_CONSUMER_BEFORE.search(source[max(0, start - 40):start])
                or _JS_CONSUMER_AFTER.match(source[end:end + 40])):
            continue
        for match in LINE_OPENING_MARKER_RE.finditer(text):
            seen.setdefault(match.group(1), set()).add((label, ""))
        for match in PARTIAL_MARKER_RE.finditer(text):
            line = source.count("\n", 0, start) + 1
            constructions.add(Construction(label, line, match.group(0).replace(HOLE, "{…}")))
    return seen, constructions


#: The last few scans, keyed by the sources' exact text: the tests read the real
#: tree several times, and a scratch tree a test edits in place is read afresh.
_SCAN_CACHE: Dict[Tuple[str, str], Tuple[Dict[str, List[Emission]], List[Construction]]] = {}


def _scan(skill_root: Path) -> Tuple[Dict[str, List[Emission]], List[Construction]]:
    sources = [(path, label, path.read_text(encoding="utf-8"))
               for path, label in _source_files(skill_root)]
    digest = hashlib.sha256()
    for _, label, text in sources:
        digest.update(f"{label}\0{text}\0".encode("utf-8"))
    key = (str(skill_root.resolve()), digest.hexdigest())
    if key not in _SCAN_CACHE:
        modules = [_Module(path, label, text) for path, label, text in sources
                   if path.suffix == ".py"]
        emitted, constructions = _Scanner(modules).scan()
        found = set(constructions)
        for path, label, text in sources:
            if path.suffix == ".py":
                continue
            seen, built = _scan_js(label, text)
            found |= built
            for marker, places in seen.items():
                rows = emitted.setdefault(marker, [])
                rows.extend(Emission(p, c, "") for p, c in sorted(places)
                            if all(r.path != p for r in rows))
                rows.sort(key=lambda e: (e.path, e.constant))
        while len(_SCAN_CACHE) >= 4:
            _SCAN_CACHE.pop(next(iter(_SCAN_CACHE)))
        _SCAN_CACHE[key] = (emitted, sorted(found))
    emitted, constructions = _SCAN_CACHE[key]
    return {marker: list(rows) for marker, rows in emitted.items()}, list(constructions)


def scan_emissions(skill_root: Path = SKILL_ROOT) -> Dict[str, List[Emission]]:
    """Every marker a produced string opens a line with, and where it is fixed."""
    return _scan(skill_root)[0]


def scan_constructions(skill_root: Path = SKILL_ROOT) -> List[Construction]:
    """Printed brackets built from upper-case text and a run-time value."""
    return _scan(skill_root)[1]


def citing_files(skill_root: Path = SKILL_ROOT) -> List[Path]:
    files = [skill_root / "SKILL.md"]
    files.extend(sorted(skill_root.glob(CITING_SHIMS)))
    for directory in CITING_DIRS:
        files.extend(sorted((skill_root / directory).glob("*.md")))
    return [path for path in files if path.is_file()]


def scan_citations(skill_root: Path = SKILL_ROOT) -> Dict[str, List[str]]:
    """Marker -> the instruction, reference, SKILL.md or shim files that cite it."""
    cited: Dict[str, Set[str]] = {}
    for path in citing_files(skill_root):
        relative = path.relative_to(skill_root).as_posix()
        for marker in MARKER_RE.findall(path.read_text(encoding="utf-8")):
            cited.setdefault(marker, set()).add(relative)
    return {marker: sorted(files) for marker, files in cited.items()}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_HEADER = """<!-- GENERATED FILE — DO NOT EDIT.
     Rendered by skills/multi-llm/tests/tools/render_markers.py from the skill's
     Python sources, the hand-written packaging/ sources and the skill's SKILL.md,
     host shim templates, instructions/ and references/. Regenerate with
     `uv run --project skills/multi-llm -- python skills/multi-llm/tests/tools/render_markers.py`;
     `tests/test_markers_registry.py` fails when this file differs from a fresh render. -->

# Bracket markers

The orchestrators print lines that start with a bracket marker such as `[BATCH_DONE]`.
Hosts and the opencode and codex servers parse some of them, and the instructions tell
the driving agent what to do on others, so **a marker is a contract: never rename one**
without changing every file in its row.

Four tests keep this page honest (`tests/test_markers_registry.py`):

- the page matches a fresh render of the sources;
- every marker `SKILL.md`, a host shim template, `instructions/` or `references/` cites is emitted by a
  source;
- every emitted marker is either cited by one of those documents or listed as
  not-for-the-host in `NOT_FOR_HOST` in the generator, with a reason. The page cannot
  tell a line the host must act on from one it only relays, so every marker counts as
  host-facing until it is listed;
- no printed marker is built from upper-case text and a run-time value
  (`f"[BATCH_{kind}]"`): write each one whole, as a literal or a constant, so the scan
  can list it.

**What counts as emitted.** The sources are the skill's non-test `.py` files and the
hand-written `packaging/` sources, never the generated `packaging/*/multi-llm/` copies.
A marker is emitted where a string the code prints, writes, appends, returns or stores
opens a line with it. A string the code only tests for the marker (`in`, `startswith`,
a regex) is a parser, and a sentence that names a marker mid-line mentions it; neither
emits it. Strings are followed through f-strings, `+`, constants, imports, local names
and a helper's parameters (`sweep_live_clis("DEADLINE_SWEEP")`). The generator's
docstring has the full rules.

**How to read a row.** *Emitted by* names a file (under `skills/multi-llm/`, or
`packaging/…` from the repository root) and, where the marker's text comes from a
module constant or a helper's argument, the constant or `helper()` to grep for; there
are no line numbers because they would make the page stale after any edit above them.
*Consumed by* is where an instruction, reference, `SKILL.md` or shim template cites the marker; where
none does, it shows the reason key from the legend below. *Meaning* is the first
sentence of the comment or docstring on the constant, and `—` where the marker is an
inline literal or has none. Cells list a few entries and then `+N more`; grep for the
rest. A marker that only a comment, a docstring or a parser mentions is not emitted
and is not listed.

To add a marker: emit it, then either cite it in the instruction that handles it or add
it to `NOT_FOR_HOST` with a reason, then re-render this page.
"""


def _emitters_cell(emissions: Sequence[Emission]) -> str:
    items = [f"{e.path} {e.constant}".strip() if e.constant else e.path for e in emissions]
    shown = []
    for item in items[:MAX_EMITTERS]:
        path, _, constant = item.partition(" ")
        shown.append(f"`{path}`" + (f" `{constant}`" if constant else ""))
    if len(items) > MAX_EMITTERS:
        shown.append(f"+{len(items) - MAX_EMITTERS} more")
    return "<br>".join(shown)


def render(skill_root: Path = SKILL_ROOT) -> str:
    emitted = scan_emissions(skill_root)
    cited = scan_citations(skill_root)
    rows: List[str] = []
    for marker in sorted(emitted):
        emissions = emitted[marker]
        if marker in cited:
            consumer = "<br>".join(f"`{p}`" for p in cited[marker][:MAX_CONSUMERS])
            if len(cited[marker]) > MAX_CONSUMERS:
                consumer += f"<br>+{len(cited[marker]) - MAX_CONSUMERS} more"
        elif marker in NOT_FOR_HOST:
            consumer = f"not for the host: {NOT_FOR_HOST[marker]}"
        else:
            consumer = "**unhandled**"
        meaning = next((e.meaning for e in emissions if e.meaning), "") or "—"
        rows.append(f"| `{marker}` | {_emitters_cell(emissions)} | {consumer} | {meaning} |")

    host_handled = sum(1 for m in emitted if m in cited)
    unhandled = sum(1 for m in emitted if m not in cited and m not in NOT_FOR_HOST)
    out = [_HEADER, "## Counts", ""]
    out.append(f"- Emitted markers: {len(emitted)}")
    out.append(f"- Cited by an instruction, reference, `SKILL.md` or shim template: {host_handled}")
    out.append(f"- Declared not-for-the-host: {len([m for m in emitted if m in NOT_FOR_HOST and m not in cited])}")
    out.append(f"- Neither (must be zero): {unhandled}")
    out += ["", "## Registry", "", "| Marker | Emitted by | Consumed by | Meaning |",
            "| --- | --- | --- | --- |"]
    out += rows
    out += ["", "## Not-for-the-host reasons", "", "| Key | Why no instruction names the marker |",
            "| --- | --- |"]
    out += [f"| `{key}` | {text} |" for key, text in REASONS.items()]
    return "\n".join(out) + "\n"


def check(output: Optional[Path] = None, skill_root: Path = SKILL_ROOT) -> Optional[str]:
    """``None`` when ``output`` is a fresh render, else a message that names the fix."""
    output = OUTPUT if output is None else output
    fix = "run: uv run --project skills/multi-llm -- python skills/multi-llm/tests/tools/render_markers.py"
    if not output.is_file():
        return f"{output} is missing; {fix}"
    fresh = render(skill_root)
    current = output.read_text(encoding="utf-8")
    if current == fresh:
        return None
    diff = "".join(list(difflib.unified_diff(
        current.splitlines(True), fresh.splitlines(True), "committed", "fresh", n=0))[:40])
    return f"{output} is stale; {fix}\n{diff}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the page is stale")
    parser.add_argument("--stdout", action="store_true", help="print the page, write nothing")
    args = parser.parse_args(argv)
    if args.check:
        problem = check()
        if problem:
            print(problem, file=sys.stderr)
            return 1
        print("docs/development/markers.md is up to date")
        return 0
    text = render()
    if args.stdout:
        sys.stdout.write(text)
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
