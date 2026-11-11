"""The rules ``tests/support/`` keeps, from its ``__init__.py``. Static and zero-spend.

A module ``tests/conftest.py`` registers (``pytest_plugins``) gives its fixtures
to every test file, so it defines no autouse fixture and no fixture name that
another registered module, or the conftest itself, defines; and it imports no
fixture, which would register that one suite-wide too. A support module that is
not registered gives a fixture only to the files that import it, so each of its
fixtures carries ``name=`` and a function name of its own: importing the function
then binds no module-level name a test parameter would shadow (ruff F811).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import pytest

from tests import conftest

TESTS = Path(__file__).resolve().parent
SUPPORT = TESTS / "support"
PREFIX = "tests.support."


class Fixture(NamedTuple):
    function: str
    name: str  # the name a test requests: ``name=`` or the function's
    explicit: bool  # the decorator carries ``name=``
    autouse: bool


def _fixture_decorator(node: ast.expr) -> Optional[ast.expr]:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute) and target.attr == "fixture":
        return node
    if isinstance(target, ast.Name) and target.id == "fixture":
        return node
    return None


def _keyword(call: ast.expr, key: str):
    if isinstance(call, ast.Call):
        for kw in call.keywords:
            if kw.arg == key and isinstance(kw.value, ast.Constant):
                return kw.value.value
    return None


def fixtures(path: Path) -> List[Fixture]:
    """The top-level fixtures *path* defines."""
    found = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if _fixture_decorator(dec) is not None:
                name = _keyword(dec, "name")
                found.append(Fixture(node.name, name or node.name, name is not None,
                                     bool(_keyword(dec, "autouse"))))
    return found


def support_modules() -> Dict[str, Path]:
    return {f"{PREFIX}{p.stem}": p for p in sorted(SUPPORT.glob("*.py")) if p.stem != "__init__"}


REGISTERED = list(conftest.pytest_plugins)
SUPPORT_MODULES = support_modules()
#: The support modules not registered that define fixtures (each one's tests import them).
UNREGISTERED = sorted(name for name in set(SUPPORT_MODULES) - set(REGISTERED)
                      if fixtures(SUPPORT_MODULES[name]))


def test_the_registered_modules_are_support_modules():
    assert REGISTERED == sorted(REGISTERED), "keep pytest_plugins sorted"
    for name in REGISTERED:
        assert name in SUPPORT_MODULES, f"{name} is not a tests/support module"
        assert fixtures(SUPPORT_MODULES[name]), f"{name} defines no fixture to register"


@pytest.mark.parametrize("name", REGISTERED)
def test_a_registered_module_has_no_autouse_fixture(name):
    autouse = [f.function for f in fixtures(SUPPORT_MODULES[name]) if f.autouse]
    assert not autouse, f"{name}: autouse fixtures would run for every test: {autouse}"


def test_no_fixture_name_is_registered_twice():
    seen: Dict[str, str] = {f.name: "tests/conftest.py" for f in fixtures(TESTS / "conftest.py")}
    for name in REGISTERED:
        for fixture in fixtures(SUPPORT_MODULES[name]):
            assert fixture.name not in seen, (
                f"{name} registers `{fixture.name}`, which {seen[fixture.name]} defines too")
            seen[fixture.name] = name


@pytest.mark.parametrize("name", REGISTERED)
def test_a_registered_module_imports_no_fixture(name):
    tree = ast.parse(SUPPORT_MODULES[name].read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(PREFIX):
            source = SUPPORT_MODULES.get(node.module)
            defined = {f.function for f in fixtures(source)} if source else set()
            leaked = [a.name for a in node.names if a.name in defined]
            assert not leaked, f"{name} imports fixtures {leaked}, registering them suite-wide"


@pytest.mark.parametrize("name", UNREGISTERED)
def test_an_unregistered_modules_fixtures_are_named_apart_from_their_functions(name):
    plain = [f.function for f in fixtures(SUPPORT_MODULES[name])
             if not f.explicit or f.name == f.function]
    assert not plain, (
        f"{name}: give {plain} `@pytest.fixture(name=...)` and a function name of their own "
        "(or register the module in tests/conftest.py)")


def test_no_test_file_imports_a_registered_fixture_under_its_own_name():
    """It is suite-wide already; an import binds a module-level copy that every
    test parameter of that name shadows. An import under another name
    (``repo as gate_repo``) is a fixture of that other name, and allowed."""
    registered = {name: {f.function for f in fixtures(SUPPORT_MODULES[name])} for name in REGISTERED}
    found = []
    for path in sorted(TESTS.glob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module in registered:
                found += [f"{path.name}:{node.lineno} {a.name}" for a in node.names
                          if a.name in registered[node.module] and not a.asname]
    assert not found, found
