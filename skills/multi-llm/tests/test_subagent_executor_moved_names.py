"""The contract for the names `utils.subagent_executor` re-exports from the
thirteen modules split out of it, and from `known_checks` (stated in the
executor's module docstring). The executor is a facade: it defines nothing
but `__all__` and its re-export block.

* **Reads work.** `subagent_executor.X`, `from utils.subagent_executor import
  X`, `dir()`, `inspect.getmembers` and the executor's `__all__` see every
  moved name, as its owner's current object.
* **Writes do not.** A patch or assignment of any name on the executor only
  shadows it there and reaches no code, because no code is left there. The
  guard at the end fails any test that writes one: a moved name names its
  owner, an imported one the modules that read it.
* **Every reader goes through the owner at each call.** The modules spell a
  moved name `_<owner>.X`, never a copy, so one patch of the owner reaches
  the dispatch and every other module.
* **The modules import down only**: `batch_reply`, `failure_capture`,
  `task_outcome`, `task_result`, `runner_selection`, `probe_state`,
  `sandbox_walk`, `dispatch_spawn`, `attempt_classifier`, `runner_preflight`,
  `task_dispatch`, `task_batch`, `task_file_runner`; none of them imports the
  executor, and `known_checks` imports none of them (the cycle EXEC-5 step 3
  broke).
"""

import ast
import inspect
import symtable
import sys
from pathlib import Path
from unittest import mock

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from tests.moved_name_writes import MovedNameWriteScanner  # noqa: E402
from utils import (  # noqa: E402
    attempt_classifier,
    batch_reply,
    dispatch_spawn,
    failure_capture,
    known_checks,
    probe_state,
    runner_preflight,
    runner_selection,
    sandbox_walk,
    subagent_executor,
    task_batch,
    task_dispatch,
    task_file_runner,
    task_outcome,
    task_result,
)

#: Lowest first: each module may import only the ones before it.
OWNERS = {
    "batch_reply": batch_reply,
    "failure_capture": failure_capture,
    "task_outcome": task_outcome,
    "task_result": task_result,
    "runner_selection": runner_selection,
    "probe_state": probe_state,
    "sandbox_walk": sandbox_walk,
    "dispatch_spawn": dispatch_spawn,
    "attempt_classifier": attempt_classifier,
    "runner_preflight": runner_preflight,
    "task_dispatch": task_dispatch,
    "task_batch": task_batch,
    "task_file_runner": task_file_runner,
}
#: An existing module that owns one moved name; it keeps its own names too.
KNOWN_CHECKS_MOVED = ("CHECK_FAILED_WITHOUT_FINDINGS",)
ALL_OWNERS = {"known_checks": known_checks, **OWNERS}
EXECUTOR = "subagent_executor"
#: All the executor defines.
FACADE = {"__all__", "_MOVED", "_MOVED_TO", "__getattr__", "__dir__"}


def _source(module_name):
    return (SKILL_ROOT / "utils" / f"{module_name}.py").read_text(encoding="utf-8")


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


def _imported_modules(module_name):
    """The `utils` modules *module_name* imports, at module level or inside a
    function, by their last dotted part."""
    found = set()
    for node in ast.walk(ast.parse(_source(module_name))):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                found.add(node.module.split(".")[-1])
            elif node.level:
                found.update(alias.name for alias in node.names)
            elif node.module == "utils":
                found.update(alias.name for alias in node.names)
            elif node.module and node.module.startswith("utils"):
                found.add(node.module.split(".")[-1])
        elif isinstance(node, ast.Import):
            found.update(alias.name.split(".")[-1] for alias in node.names
                         if alias.name.startswith("utils."))
    return found


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class TestTheMovedNamesTable:
    @pytest.mark.parametrize("owner", list(OWNERS))
    def test_it_lists_exactly_what_each_new_module_defines(self, owner):
        assert set(subagent_executor._MOVED[owner]) == _top_level_definitions(owner)

    def test_known_checks_owns_the_one_constant(self):
        assert subagent_executor._MOVED["known_checks"] == KNOWN_CHECKS_MOVED
        assert set(KNOWN_CHECKS_MOVED) <= _top_level_definitions("known_checks")

    def test_it_names_the_owners(self):
        assert list(subagent_executor._MOVED) == list(ALL_OWNERS)

    def test_the_executor_defines_and_binds_none_of_them(self):
        moved = set(subagent_executor._MOVED_TO)
        assert not _top_level_definitions(EXECUTOR) & moved
        assert not set(vars(subagent_executor)) & moved

    def test_the_executor_defines_nothing_but_the_facade(self):
        assert _top_level_definitions(EXECUTOR) == FACADE

    def test_every_name_in_all_still_reads(self):
        for name in subagent_executor.__all__:
            assert hasattr(subagent_executor, name), name


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    def test_every_moved_name_reads_as_its_owners_object(self):
        for name, owner in subagent_executor._MOVED_TO.items():
            assert getattr(subagent_executor, name) is getattr(ALL_OWNERS[owner], name), name

    def test_from_import(self):
        namespace = {}
        exec(
            "from utils.subagent_executor import TaskOutcome, "
            "check_findings_from, sandbox_verdict, CHECK_FAILED_WITHOUT_FINDINGS",
            namespace,
        )
        assert namespace["TaskOutcome"] is task_outcome.TaskOutcome
        assert namespace["check_findings_from"] is batch_reply.check_findings_from
        assert namespace["sandbox_verdict"] is sandbox_walk.sandbox_verdict
        assert namespace["CHECK_FAILED_WITHOUT_FINDINGS"] is (
            known_checks.CHECK_FAILED_WITHOUT_FINDINGS)

    def test_a_cache_reads_as_the_registered_object(self):
        """The registry clears the object the owner registered; a read through
        the executor must be that same object, not a copy."""
        assert subagent_executor._SANDBOX_PROBE_CACHE is probe_state._SANDBOX_PROBE_CACHE
        assert subagent_executor._VERSION_CACHE is probe_state._VERSION_CACHE

    def test_a_patch_on_the_owner_shows_through_the_executor(self):
        with mock.patch.object(sandbox_walk, "sandbox_verdict") as fake:
            assert subagent_executor.sandbox_verdict is fake
        assert subagent_executor.sandbox_verdict is sandbox_walk.sandbox_verdict

    def test_an_unknown_name_is_an_attribute_error_naming_the_executor(self):
        with pytest.raises(
            AttributeError,
            match=r"'utils\.subagent_executor' has no attribute '_u44_absent'",
        ):
            subagent_executor._u44_absent
        assert not hasattr(subagent_executor, "_u44_absent")

    def test_dir_and_getmembers_list_them(self):
        assert set(subagent_executor._MOVED_TO) <= set(dir(subagent_executor))
        members = dict(inspect.getmembers(subagent_executor))
        for name, owner in subagent_executor._MOVED_TO.items():
            assert members[name] is getattr(ALL_OWNERS[owner], name), name


# ---------------------------------------------------------------------------
# Every reader goes through the owner
# ---------------------------------------------------------------------------


def _global_reads(source):
    """The names *source* reads from its module's global scope, in any scope:
    a local of the same name (a parameter, an assignment) is not one."""
    found = set()

    def walk(scope):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and (
                scope.get_type() == "module" or symbol.is_global()
            ):
                found.add(symbol.get_name())
        for child in scope.get_children():
            walk(child)

    walk(symtable.symtable(source, "<module>", "exec"))
    return found


def _qualified_reads(module_name):
    """``(alias, name)`` for every ``_<owner>.<name>`` in *module_name*."""
    aliases = {f"_{owner}" for owner in ALL_OWNERS}
    return [
        (node.value.id, node.attr) for node in ast.walk(ast.parse(_source(module_name)))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    ]


class TestEveryReaderGoesThroughTheOwner:
    @pytest.mark.parametrize("module_name", [EXECUTOR, *OWNERS])
    def test_no_moved_name_is_read_bare_or_copied(self, module_name):
        """A moved name defined elsewhere is never a bare name here and never
        imported by name: each read spells `_<owner>.X`."""
        own = _top_level_definitions(module_name) if module_name in OWNERS else set()
        foreign = set(subagent_executor._MOVED_TO) - own
        tree = ast.parse(_source(module_name))
        bare = sorted(_global_reads(_source(module_name)) & foreign)
        copied = sorted({
            alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if (node.module or "").split(".")[-1] in OWNERS or alias.name in foreign
        })
        assert bare == [] and copied == [], (bare, copied)

    @pytest.mark.parametrize("module_name", [EXECUTOR, *OWNERS])
    def test_every_qualified_read_names_what_its_owner_defines(self, module_name):
        for alias, name in _qualified_reads(module_name):
            assert name in _top_level_definitions(alias[1:]), f"{alias}.{name}"

    def test_the_modules_read_their_owners_through_qualified_names(self):
        assert len(_qualified_reads("task_dispatch")) > 40
        assert _qualified_reads(EXECUTOR) == []

    def test_a_patch_on_the_owner_reaches_the_code_that_reads_it(self):
        with mock.patch.object(task_outcome, "_batch_report_filename",
                               return_value="pinned.json") as fake:
            note = subagent_executor._pinned_runner_note(
                "codex:m" + runner_selection._RECORDED_RUNNER_SUFFIX, "validation")
        assert "kept from pinned.json" in note
        fake.assert_called_once_with("validation")

    def test_a_patch_on_the_executor_reaches_none_of_it(self):
        """The unsupported write the guard below refuses, pinned to its
        documented effect: a shadow in the executor that no code reads."""
        with mock.patch.object(subagent_executor, "_batch_report_filename",
                               return_value="never.json", create=True) as fake:
            note = subagent_executor._pinned_runner_note(
                "codex:m" + runner_selection._RECORDED_RUNNER_SUFFIX, "validation")
        assert "never.json" not in note and "kept from" in note
        assert fake.call_count == 0
        assert "_batch_report_filename" not in vars(subagent_executor)
        assert subagent_executor._batch_report_filename is task_outcome._batch_report_filename

    def test_a_patch_of_an_imported_name_on_the_executor_reaches_none_of_it(self):
        """The executor keeps a few imports for readers (``subprocess``,
        ``RetryLadder``); patching one there reaches no dispatch, the module
        whose code reads it is the one to patch."""
        assert subagent_executor.RetryLadder is task_dispatch.RetryLadder
        with mock.patch.object(subagent_executor, "RetryLadder") as fake:
            assert task_dispatch.RetryLadder is not fake


# ---------------------------------------------------------------------------
# The layering
# ---------------------------------------------------------------------------


class TestTheLayering:
    @pytest.mark.parametrize("module_name", list(OWNERS))
    def test_a_new_module_imports_only_the_ones_below_it(self, module_name):
        order = list(OWNERS)
        above = set(order[order.index(module_name) + 1:]) | {EXECUTOR}
        assert not _imported_modules(module_name) & above

    def test_known_checks_imports_neither_the_executor_nor_its_modules(self):
        assert not _imported_modules("known_checks") & ({EXECUTOR} | set(OWNERS))

    def test_the_executor_imports_all_of_them(self):
        assert set(ALL_OWNERS) <= _imported_modules(EXECUTOR)

    def test_the_dispatch_is_below_the_batch_and_the_entry_points(self):
        """``task_batch`` runs ``task_dispatch.dispatch_task``, the entry
        points run ``task_batch.run_tasks``; the dispatch reads neither."""
        assert "task_dispatch" in _imported_modules("task_batch")
        assert "task_batch" in _imported_modules("task_file_runner")
        assert not _imported_modules("task_dispatch") & {"task_batch", "task_file_runner"}


# ---------------------------------------------------------------------------
# The guard: no test writes a moved name on the executor
# ---------------------------------------------------------------------------


def _guard_message(path, lineno, form, name, owner):
    if owner is None:
        readers = sorted(m for m in OWNERS if name in _imported_names(m))
        return (
            f"{path}:{lineno}: {form} writes `{name}` on utils.subagent_executor, "
            f"which has no code left to read it: the executor only re-exports "
            f"names for reading. Patch {name} in the module whose code your test "
            f"drives (it is imported by: {', '.join(readers) or 'none of them'})."
        )
    return (
        f"{path}:{lineno}: {form} writes `{name}` on utils.subagent_executor, but "
        f"{name} lives in utils/{owner}.py and the executor only re-exports it for "
        f"reading: the code that uses it never sees this patch. Patch "
        f"utils.{owner}.{name} instead."
    )


def _imported_names(module_name):
    return {
        alias.asname or alias.name
        for node in ast.walk(ast.parse(_source(module_name)))
        if isinstance(node, ast.ImportFrom) for alias in node.names
    }


class _AnyName(dict):
    """Every name, for the scanner: a moved one maps to its owner, any other
    name the executor binds or a test invents maps to None."""

    def __contains__(self, name):
        return name is not None

    def __getitem__(self, name):
        return subagent_executor._MOVED_TO.get(name)


def test_no_test_writes_any_name_on_the_executor():
    """This file is the one exception: it exercises those writes on purpose."""
    scanner = MovedNameWriteScanner(_AnyName(), EXECUTOR)
    tests_dir = Path(__file__).parent
    problems = [
        _guard_message(path.relative_to(SKILL_ROOT), *hit)
        for path in sorted(tests_dir.rglob("*.py"))
        if path != Path(__file__)
        for hit in scanner.scan(path.read_text(encoding="utf-8"), str(path))
    ]
    assert problems == [], "\n".join(problems)


_IMPORTS = "from unittest import mock\nfrom utils import subagent_executor\n"


class TestTheGuardSeesTheExecutor:
    @pytest.mark.parametrize("snippet, name", [
        ('monkeypatch.setattr(subagent_executor, "_run_sandbox_probe", None)',
         "_run_sandbox_probe"),
        ('monkeypatch.setattr("utils.subagent_executor.SANDBOX_PROBE_TIMEOUT_S", 1)',
         "SANDBOX_PROBE_TIMEOUT_S"),
        ('from utils import subagent_executor as executor\n'
         'mock.patch.object(executor, "sandbox_verdict")', "sandbox_verdict"),
        ('import utils.subagent_executor as ex\n'
         'ex.MAX_RESULT_BYTES = 1', "MAX_RESULT_BYTES"),
        ('mock.patch("utils.subagent_executor._redact_probe_tail")', "_redact_probe_tail"),
        ('mock.patch(target="utils.subagent_executor._atomic_write_json")',
         "_atomic_write_json"),
        ('monkeypatch.setattr(target="utils.subagent_executor.persist_result", name=None)',
         "persist_result"),
        ('monkeypatch.setattr(subagent_executor, name="_fallback_rungs", value=None)',
         "_fallback_rungs"),
        ('mock.patch.object(target=subagent_executor, attribute="write_rung_notice")',
         "write_rung_notice"),
        ('monkeypatch.setattr(subagent_executor, "CHECK_FAILED_WITHOUT_FINDINGS", "x")',
         "CHECK_FAILED_WITHOUT_FINDINGS"),
        ('monkeypatch.setattr(subagent_executor, "_spawn", None)', "_spawn"),
        ('monkeypatch.setattr(subagent_executor, "_run_probe", None)', "_run_probe"),
        ('mock.patch("utils.subagent_executor.dispatch_task")', "dispatch_task"),
        ('monkeypatch.setattr(subagent_executor, "WRITE_SCOPE_BY_KIND", {})',
         "WRITE_SCOPE_BY_KIND"),
    ])
    def test_a_write(self, snippet, name):
        scanner = MovedNameWriteScanner(subagent_executor._MOVED_TO, EXECUTOR)
        hits = scanner.scan(_IMPORTS + snippet)
        assert [hit[2] for hit in hits] == [name]
        owner = subagent_executor._MOVED_TO[name]
        assert f"Patch utils.{owner}.{name} instead" in _guard_message("t.py", *hits[0])

    @pytest.mark.parametrize("snippet, name", [
        ('monkeypatch.setattr(subagent_executor, "get_provider", None)', "get_provider"),
        ('mock.patch.object(subagent_executor, "RetryLadder")', "RetryLadder"),
        ('mock.patch("utils.subagent_executor.subprocess")', "subprocess"),
    ])
    def test_a_write_of_an_imported_name(self, snippet, name):
        scanner = MovedNameWriteScanner(_AnyName(), EXECUTOR)
        hits = scanner.scan(_IMPORTS + snippet)
        assert [(hit[2], hit[3]) for hit in hits] == [(name, None)]
        assert "has no code left to read it" in _guard_message("t.py", *hits[0])

    def test_an_imported_name_message_names_its_readers(self):
        message = _guard_message("t.py", 1, "setattr(...)", "get_subagent_timeout", None)
        assert "task_batch" in message and "task_dispatch" in message

    @pytest.mark.parametrize("snippet", [
        'monkeypatch.setattr(dispatch_spawn, "_spawn", None)',
        'monkeypatch.setattr(runner_preflight, "_run_probe", None)',
        'monkeypatch.setattr(sandbox_walk, "_run_sandbox_probe", None)',
        'monkeypatch.setattr(subagent_executor.subprocess, "Popen", None)',
        'mock.patch("utils.sandbox_walk.sandbox_verdict")',
        'subagent_executor._SANDBOX_PROBE_CACHE.clear()',
        'found = subagent_executor.check_findings_from({})',
    ])
    def test_not_a_write(self, snippet):
        scanner = MovedNameWriteScanner(_AnyName(), EXECUTOR)
        assert scanner.scan(_IMPORTS + snippet) == []
