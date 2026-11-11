"""The contract for the names `utils.run_status` re-exports from the four
`status_*` modules split out of it (stated in run_status' module docstring):

* **Reads work.** `run_status.X`, `from utils.run_status import X`, `dir()` and
  `inspect.getmembers` see every moved name, as its owner's current object.
* **Writes do not.** A patch or assignment of a moved name on `run_status` only
  shadows it there and reaches no code. The guard at the end fails any test
  that writes one, naming the owner to patch.
* **The seams tests patch stay in `run_status`**: the collector the wait loop
  re-reads, the wait itself, its clock, the next-step emitter, `--abort`.
* **The workflow read is one seam.** Every reader calls
  `status_views._workflow_payload` on its owner at each call, never a copy, so
  one patch there reaches them all.
* **The modules import down only**: views, then commands, native, text, and
  none of them imports `run_status`.
"""

import ast
import inspect
import sys
from pathlib import Path
from unittest import mock

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from tests.moved_name_writes import MovedNameWriteScanner  # noqa: E402
from utils import (  # noqa: E402
    run_status,
    status_commands,
    status_native,
    status_text,
    status_views,
)

#: Lowest first: each module may import only the ones before it.
OWNERS = {
    "status_views": status_views,
    "status_commands": status_commands,
    "status_native": status_native,
    "status_text": status_text,
}

#: The run_status names tests patch to reach `main`: they stay defined there.
KEPT_SEAMS = frozenset({
    "_collect", "_emit_user_next_step", "_resolve_wait_timeout", "_run_abort",
    "_wait_for_terminal", "note_swallowed", "time",
})


def _source(module_name):
    return (SKILL_ROOT / "utils" / f"{module_name}.py").read_text()


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
    @pytest.mark.parametrize("owner", sorted(OWNERS))
    def test_it_lists_exactly_what_each_owner_defines(self, owner):
        assert set(run_status._MOVED[owner]) == _top_level_definitions(owner)

    def test_run_status_defines_and_binds_none_of_them(self):
        assert not _top_level_definitions("run_status") & set(run_status._MOVED_TO)
        assert not set(vars(run_status)) & set(run_status._MOVED_TO)

    def test_the_seams_tests_patch_stay_in_run_status(self):
        assert not KEPT_SEAMS & set(run_status._MOVED_TO)
        assert KEPT_SEAMS <= set(vars(run_status))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    def test_every_moved_name_reads_as_its_owners_object(self):
        for name, owner in run_status._MOVED_TO.items():
            assert getattr(run_status, name) is getattr(OWNERS[owner], name), name

    def test_from_import(self):
        namespace = {}
        exec("from utils.run_status import _mirror_text, discover_phase_dirs", namespace)
        assert namespace["_mirror_text"] is status_text._mirror_text
        assert namespace["discover_phase_dirs"] is status_views.discover_phase_dirs

    def test_a_patch_on_the_owner_shows_through_run_status(self):
        with mock.patch.object(status_native, "superseded_by_native") as fake:
            assert run_status.superseded_by_native is fake
        assert run_status.superseded_by_native is status_native.superseded_by_native

    def test_an_unknown_name_is_an_attribute_error_naming_run_status(self):
        with pytest.raises(AttributeError, match=r"'utils\.run_status' has no attribute '_u54c_absent'"):
            run_status._u54c_absent
        assert not hasattr(run_status, "_u54c_absent")

    def test_dir_and_getmembers_list_them(self):
        assert set(run_status._MOVED_TO) <= set(dir(run_status))
        members = dict(inspect.getmembers(run_status))
        for name, owner in run_status._MOVED_TO.items():
            assert members[name] is getattr(OWNERS[owner], name), name


# ---------------------------------------------------------------------------
# The workflow read and the layering
# ---------------------------------------------------------------------------


class TestTheWorkflowReadIsOneSeam:
    @pytest.mark.parametrize("module_name", ["run_status", *OWNERS])
    def test_every_reader_calls_it_on_its_owner(self, module_name):
        """No bare `_workflow_payload` outside its owner and no copy of it:
        each reader spells `status_views._workflow_payload`."""
        tree = ast.parse(_source(module_name))
        bare = [node.lineno for node in ast.walk(tree)
                if isinstance(node, ast.Name) and node.id == "_workflow_payload"]
        copied = [node.lineno for node in ast.walk(tree)
                  if isinstance(node, ast.ImportFrom)
                  and any(alias.name == "_workflow_payload" for alias in node.names)]
        if module_name == "status_views":
            assert copied == []
        else:
            assert bare == [] and copied == [], (bare, copied)

    def test_a_patch_there_reaches_a_reader_in_another_module(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n")
        with mock.patch.object(status_views, "_workflow_payload",
                               return_value={"phases": {"implement": {"status": "in_progress"}}}) as fake:
            view = {"phase": "implement", "run": {}}
            assert status_commands._phase_in_progress(view, plan, None) is True
        assert fake.call_count == 1


class TestTheLayering:
    @pytest.mark.parametrize("module_name", sorted(OWNERS))
    def test_a_status_module_imports_only_the_ones_below_it(self, module_name):
        order = list(OWNERS)
        above = set(order[order.index(module_name) + 1:]) | {"run_status"}
        assert not _imported_modules(module_name) & above


# ---------------------------------------------------------------------------
# The guard: no test writes a moved name on run_status
# ---------------------------------------------------------------------------


def _guard_message(path, lineno, form, name, owner):
    return (
        f"{path}:{lineno}: {form} writes `{name}` on utils.run_status, but {name} lives in "
        f"utils/{owner}.py and run_status only re-exports it for reading: the code that "
        f"uses it never sees this patch. Patch utils.{owner}.{name} instead."
    )


def test_no_test_writes_a_moved_name_on_run_status():
    """This file is the one exception: it exercises those writes on purpose."""
    scanner = MovedNameWriteScanner(run_status._MOVED_TO, "run_status")
    tests_dir = Path(__file__).parent
    problems = [
        _guard_message(path.relative_to(SKILL_ROOT), *hit)
        for path in sorted(tests_dir.rglob("*.py"))
        if path != Path(__file__)
        for hit in scanner.scan(path.read_text(), str(path))
    ]
    assert problems == [], "\n".join(problems)


_IMPORTS = "from unittest import mock\nfrom utils import run_status\n"


class TestTheGuardSeesRunStatus:
    @pytest.mark.parametrize("snippet, name", [
        ('monkeypatch.setattr(run_status, "_workflow_payload", None)', "_workflow_payload"),
        ('monkeypatch.setattr("utils.run_status._mirror_text", None)', "_mirror_text"),
        ('from utils import run_status as rs\nmock.patch.object(rs, "render_run_text")',
         "render_run_text"),
        ('run_status.superseded_by_native = None', "superseded_by_native"),
    ])
    def test_a_write(self, snippet, name):
        hits = MovedNameWriteScanner(run_status._MOVED_TO, "run_status").scan(_IMPORTS + snippet)
        assert [hit[2] for hit in hits] == [name]
        owner = run_status._MOVED_TO[name]
        assert f"Patch utils.{owner}.{name} instead" in _guard_message("t.py", *hits[0])

    @pytest.mark.parametrize("snippet", [
        'monkeypatch.setattr(run_status, "_collect", None)',
        'monkeypatch.setattr(run_status, "time", None)',
        'monkeypatch.setattr(status_views, "_workflow_payload", None)',
        'found = run_status._mirror_text("subagents", {})',
    ])
    def test_not_a_write(self, snippet):
        assert MovedNameWriteScanner(run_status._MOVED_TO, "run_status").scan(_IMPORTS + snippet) == []
