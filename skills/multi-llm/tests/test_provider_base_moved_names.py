"""The contract for the names `utils.providers.base` re-exports from the seven
modules split out of it (stated in base's module docstring):

* **Reads work.** `base.X`, `from utils.providers.base import X`, `dir()` and
  `inspect.getmembers` see every moved name, as its owner's current object.
* **Writes do not.** A patch or assignment of a moved name on `base` only
  shadows it there and reaches no code. The guard at the end fails any test
  that writes one, naming the owner to patch.
* **`LLMProvider` and `DEFAULT_TIMEOUT_S` stay in `base`**, and the class's two
  default pattern sets are `failure_classifier`'s constants.
* **The modules import down only**, in the order of :data:`OWNERS`, and none of
  them imports `base` at run time.
* **The classifier is pure**: loaded alone, it brings in no `utils` module but
  the status vocabulary and the process-cache registry, both standard-library
  only (docs/development/providers.md#the-provider-base-modules).
"""

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from tests.moved_name_writes import MovedNameWriteScanner  # noqa: E402
from utils.providers import (  # noqa: E402
    base,
    child_env,
    failure_classifier,
    reasoning,
    sandbox_rungs,
    stdout_shape,
    types,
    version_gate,
)

PROVIDERS = SKILL_ROOT / "utils" / "providers"

#: Lowest first: each module may import only the ones before it.
OWNERS = {
    "types": types,
    "reasoning": reasoning,
    "failure_classifier": failure_classifier,
    "stdout_shape": stdout_shape,
    "child_env": child_env,
    "version_gate": version_gate,
    "sandbox_rungs": sandbox_rungs,
}

#: What base still defines itself.
KEPT = frozenset({"DEFAULT_TIMEOUT_S", "LLMProvider"})

#: The only `utils` modules the classifier may load: the status vocabulary and
#: the process-cache registry.
CLASSIFIER_LEAVES = frozenset({"utils.retry_ladder", "utils.process_state"})


def _source(module_name):
    return (PROVIDERS / f"{module_name}.py").read_text(encoding="utf-8")


def _top_level_definitions(module_name):
    names = set()
    for node in ast.parse(_source(module_name)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _is_type_checking(node):
    return isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test)


def _imported_modules(module_name):
    """The modules *module_name* imports at run time, anywhere in it (not under
    `if TYPE_CHECKING:`): sibling providers modules by their last dotted part,
    other `utils` modules as `utils.<name>`."""
    found = set()

    def visit(node):
        if _is_type_checking(node):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:
                if node.module:
                    found.add(node.module.split(".")[0])
                else:
                    found.update(alias.name for alias in node.names)
            elif node.level == 2 and node.module:
                found.add(f"utils.{node.module.split('.')[0]}")
            elif node.level == 0 and (node.module or "").startswith("utils."):
                parts = node.module.split(".")
                found.add(parts[2] if parts[1] == "providers" and len(parts) > 2
                          else f"utils.{parts[1]}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("utils."):
                    found.add(alias.name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(_source(module_name)))
    return found


class _Minimal(base.LLMProvider):
    """The base class with its four abstract members filled in, nothing else."""

    @property
    def name(self):
        return "u47-minimal"

    def is_available(self):
        return True

    def build_command(self, prompt, model=None, **kwargs):
        return ["true", prompt]

    def parse_output(self, stdout, stderr):
        return {"success": True, "data": stdout}


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class TestTheMovedNamesTable:
    def test_it_names_the_owners_lowest_first(self):
        assert list(base._MOVED) == list(OWNERS)

    @pytest.mark.parametrize("owner", sorted(OWNERS))
    def test_it_lists_exactly_what_each_owner_defines(self, owner):
        assert set(base._MOVED[owner]) == _top_level_definitions(owner)

    def test_base_defines_and_binds_none_of_them(self):
        assert not _top_level_definitions("base") & set(base._MOVED_TO)
        assert not set(vars(base)) & set(base._MOVED_TO)

    def test_base_keeps_the_class_and_its_timeout(self):
        assert KEPT <= _top_level_definitions("base")
        assert base.LLMProvider.__module__ == base.__name__


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    def test_every_moved_name_reads_as_its_owners_object(self):
        for name, owner in base._MOVED_TO.items():
            assert getattr(base, name) is getattr(OWNERS[owner], name), name

    def test_from_import(self):
        namespace = {}
        exec("from utils.providers.base import classify_streams, build_child_env", namespace)
        assert namespace["classify_streams"] is failure_classifier.classify_streams
        assert namespace["build_child_env"] is child_env.build_child_env

    def test_a_patch_on_the_owner_shows_through_base(self):
        with mock.patch.object(version_gate, "read_help_tokens") as fake:
            assert base.read_help_tokens is fake
        assert base.read_help_tokens is version_gate.read_help_tokens

    def test_a_patch_on_the_owner_reaches_llm_provider(self):
        """The class reads each moved name through its owner at every call."""
        with mock.patch.object(failure_classifier, "classify_streams",
                               return_value="sentinel") as fake:
            assert _Minimal().classify_failure("out", "err", 1) == "sentinel"
        assert fake.called

    def test_an_unknown_name_is_an_attribute_error_naming_base(self):
        with pytest.raises(AttributeError,
                           match=r"'utils\.providers\.base' has no attribute '_u47_absent'"):
            base._u47_absent
        assert not hasattr(base, "_u47_absent")

    def test_dir_and_getmembers_list_them(self):
        assert set(base._MOVED_TO) <= set(dir(base))
        members = dict(inspect.getmembers(base))
        for name, owner in base._MOVED_TO.items():
            assert members[name] is getattr(OWNERS[owner], name), name


class TestTheDefaultPatternSets:
    def test_the_class_attributes_are_the_classifiers_constants(self):
        assert base.LLMProvider.auth_error_patterns is failure_classifier.AUTH_ERROR_PATTERNS
        assert base.LLMProvider.rate_limit_patterns is failure_classifier.RATE_LIMIT_PATTERNS

    @pytest.mark.parametrize("text, status", [
        ("Error: not logged in", "auth_failed"),
        ("HTTP 429 Too Many Requests", "rate_limited"),
    ])
    def test_a_caller_with_no_adapter_matches_them(self, text, status):
        assert failure_classifier.classify_streams("", text, 1) == status
        assert failure_classifier.match_streams("", text, 1).status == status


# ---------------------------------------------------------------------------
# The layering, and the classifier's purity
# ---------------------------------------------------------------------------


class TestTheLayering:
    @pytest.mark.parametrize("module_name", list(OWNERS))
    def test_a_module_imports_only_the_ones_below_it(self, module_name):
        order = list(OWNERS)
        above = set(order[order.index(module_name):]) | {"base"}
        siblings = _imported_modules(module_name) & (set(order) | {"base"})
        assert not siblings & above, siblings & above

    def test_base_imports_every_owner(self):
        assert set(OWNERS) <= _imported_modules("base")


class TestTheClassifierIsPure:
    def test_its_imports_are_the_standard_library_and_two_leaves(self):
        utils_imports = {m for m in _imported_modules("failure_classifier")
                         if m.startswith("utils.") or m in set(OWNERS) | {"base"}}
        assert utils_imports == CLASSIFIER_LEAVES

    def test_loaded_alone_it_brings_in_no_other_utils_module(self):
        """A fresh interpreter, with the `utils` and `utils.providers` packages
        stubbed so their `__init__` barrels do not run: what importing the
        classifier loads is exactly it and its two leaves."""
        code = textwrap.dedent(f"""
            import importlib, os, sys, types
            root = {str(SKILL_ROOT)!r}
            sys.path.insert(0, root)
            for name in ("utils", "utils.providers"):
                package = types.ModuleType(name)
                package.__path__ = [os.path.join(root, *name.split("."))]
                sys.modules[name] = package
            importlib.import_module("utils.providers.failure_classifier")
            print("\\n".join(sorted(m for m in sys.modules if m.startswith("utils."))))
        """)
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        loaded = subprocess.run(
            [sys.executable, "-I", "-c", code], capture_output=True, text=True,
            check=True, env=env,
        ).stdout.split()
        assert set(loaded) == {"utils.providers", "utils.providers.failure_classifier",
                               *CLASSIFIER_LEAVES}


# ---------------------------------------------------------------------------
# The guard: no test writes a moved name on base
# ---------------------------------------------------------------------------


def _guard_message(path, lineno, form, name, owner):
    return (
        f"{path}:{lineno}: {form} writes `{name}` on utils.providers.base, but {name} "
        f"lives in utils/providers/{owner}.py and base only re-exports it for reading: "
        f"the code that uses it never sees this patch. Patch "
        f"utils.providers.{owner}.{name} instead."
    )


def test_no_test_writes_a_moved_name_on_base():
    """This file is the one exception: it exercises those writes on purpose."""
    scanner = MovedNameWriteScanner(base._MOVED_TO, "base")
    tests_dir = Path(__file__).parent
    problems = [
        _guard_message(path.relative_to(SKILL_ROOT), *hit)
        for path in sorted(tests_dir.rglob("*.py"))
        if path != Path(__file__)
        for hit in scanner.scan(path.read_text(encoding="utf-8"), str(path))
    ]
    assert problems == [], "\n".join(problems)


_IMPORTS = "from unittest import mock\nfrom utils.providers import base\n"


class TestTheGuardSeesBase:
    @pytest.mark.parametrize("snippet, name", [
        ('monkeypatch.setattr(base, "read_help_tokens", None)', "read_help_tokens"),
        ('monkeypatch.setattr("utils.providers.base.matches_any", None)', "matches_any"),
        ('from utils.providers import base as pbase\nmock.patch.object(pbase, "_provider_block")',
         "_provider_block"),
        ('import utils.providers.base as base_module\nbase_module.ENV_ALLOWLIST_MEASURED = {}',
         "ENV_ALLOWLIST_MEASURED"),
        ('monkeypatch.setattr(target=base, name="env_allowlist_enabled", value=None)',
         "env_allowlist_enabled"),
        ('mock.patch(target="utils.providers.base.matches_any")', "matches_any"),
        ('mock.patch.object(target=base, attribute="read_help_tokens")', "read_help_tokens"),
    ])
    def test_a_write(self, snippet, name):
        hits = MovedNameWriteScanner(base._MOVED_TO, "base").scan(_IMPORTS + snippet)
        assert [hit[2] for hit in hits] == [name]
        owner = base._MOVED_TO[name]
        assert f"Patch utils.providers.{owner}.{name} instead" in _guard_message("t.py", *hits[0])

    @pytest.mark.parametrize("snippet", [
        'monkeypatch.setattr(base.LLMProvider, "auth_error_patterns", ())',
        'monkeypatch.setattr(base, "DEFAULT_TIMEOUT_S", 5)',
        'monkeypatch.setattr(version_gate, "read_help_tokens", None)',
        'found = base.classify_streams("", "", 1)',
    ])
    def test_not_a_write(self, snippet):
        assert MovedNameWriteScanner(base._MOVED_TO, "base").scan(_IMPORTS + snippet) == []
