"""The contract for the names `utils.write_batch_scheduler` re-exports from the
helper modules split out of it (stated in the scheduler's module docstring):

* **Reads work.** `write_batch_scheduler.X`, `from utils.write_batch_scheduler
  import X`, `dir()`, `inspect.getmembers` and the scheduler's `__all__` see
  every moved name, as its owner's current object.
* **Writes do not.** A patch or assignment of a moved name on the scheduler
  only shadows it there and reaches no code. The guard at the end fails any
  test that writes one, naming the owner to patch.
* **Every reader goes through the owner at each call.** The scheduler and the
  helper modules spell a moved name `_<owner>.X`, never a copy, so one patch
  of the owner reaches the loop and every other helper.
* **The modules import down only**: `commit_write_guard`, `check_records`,
  `verification_lines`, `human_task_plan`, `stop_answers`, `batch_prompt`,
  `write_outcomes`, `write_context`, `write_verify`, `write_sweep`, `write_eta`,
  `write_reports`, `write_stop`, `batch_prompt_builder`, and none of them
  imports the scheduler.
"""

import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from tests.moved_name_writes import MovedNameWriteScanner  # noqa: E402
from utils import (  # noqa: E402
    batch_prompt,
    check_records,
    commit_write_guard,
    human_task_plan,
    stop_answers,
    verification_lines,
    write_batch_scheduler,
    write_outcomes,
    write_context,
    write_verify,
    write_sweep,
    write_eta,
    write_reports,
    write_stop,
    batch_prompt_builder,
)

#: Lowest first: each module may import only the ones before it.
OWNERS = {
    "commit_write_guard": commit_write_guard,
    "check_records": check_records,
    "verification_lines": verification_lines,
    "human_task_plan": human_task_plan,
    "stop_answers": stop_answers,
    "batch_prompt": batch_prompt,
    "write_outcomes": write_outcomes,
    "write_context": write_context,
    "write_verify": write_verify,
    "write_sweep": write_sweep,
    "write_eta": write_eta,
    "write_reports": write_reports,
    "write_stop": write_stop,
    "batch_prompt_builder": batch_prompt_builder,
}
SCHEDULER = "write_batch_scheduler"


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
    def test_it_lists_exactly_what_each_owner_defines(self, owner):
        assert set(write_batch_scheduler._MOVED[owner]) == _top_level_definitions(owner)

    def test_it_names_every_owner(self):
        assert list(write_batch_scheduler._MOVED) == list(OWNERS)

    def test_the_scheduler_defines_and_binds_none_of_them(self):
        moved = set(write_batch_scheduler._MOVED_TO)
        assert not _top_level_definitions(SCHEDULER) & moved
        assert not set(vars(write_batch_scheduler)) & moved

    def test_every_name_in_all_still_reads(self):
        for name in write_batch_scheduler.__all__:
            assert hasattr(write_batch_scheduler, name), name


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReads:
    def test_every_moved_name_reads_as_its_owners_object(self):
        for name, owner in write_batch_scheduler._MOVED_TO.items():
            assert getattr(write_batch_scheduler, name) is getattr(OWNERS[owner], name), name

    def test_from_import(self):
        namespace = {}
        exec(
            "from utils.write_batch_scheduler import HumanTaskPlan, "
            "split_accepted_findings, with_result_contract",
            namespace,
        )
        assert namespace["HumanTaskPlan"] is human_task_plan.HumanTaskPlan
        assert namespace["split_accepted_findings"] is check_records.split_accepted_findings
        assert namespace["with_result_contract"] is batch_prompt.with_result_contract

    def test_a_patch_on_the_owner_shows_through_the_scheduler(self):
        with mock.patch.object(stop_answers, "compose_stop_answers") as fake:
            assert write_batch_scheduler.compose_stop_answers is fake
        assert write_batch_scheduler.compose_stop_answers is stop_answers.compose_stop_answers

    def test_an_unknown_name_is_an_attribute_error_naming_the_scheduler(self):
        with pytest.raises(
            AttributeError,
            match=r"'utils\.write_batch_scheduler' has no attribute '_u42a_absent'",
        ):
            write_batch_scheduler._u42a_absent
        assert not hasattr(write_batch_scheduler, "_u42a_absent")

    def test_dir_and_getmembers_list_them(self):
        assert set(write_batch_scheduler._MOVED_TO) <= set(dir(write_batch_scheduler))
        members = dict(inspect.getmembers(write_batch_scheduler))
        for name, owner in write_batch_scheduler._MOVED_TO.items():
            assert members[name] is getattr(OWNERS[owner], name), name


# ---------------------------------------------------------------------------
# Every reader goes through the owner
# ---------------------------------------------------------------------------


def _qualified_reads(module_name):
    """``(alias, name)`` for every ``_<owner>.<name>`` in *module_name*."""
    aliases = {f"_{owner}" for owner in OWNERS}
    return [
        (node.value.id, node.attr) for node in ast.walk(ast.parse(_source(module_name)))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    ]


class TestEveryReaderGoesThroughTheOwner:
    @pytest.mark.parametrize("module_name", [SCHEDULER, *OWNERS])
    def test_no_moved_name_is_read_bare_or_copied(self, module_name):
        """A moved name defined elsewhere is never a bare name here and never
        imported by name: each read spells `_<owner>.X`."""
        own = _top_level_definitions(module_name) if module_name in OWNERS else set()
        foreign = set(write_batch_scheduler._MOVED_TO) - own
        tree = ast.parse(_source(module_name))
        bare = sorted({
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in foreign
        })
        copied = sorted({
            alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if (node.module or "").split(".")[-1] in OWNERS or alias.name in foreign
        })
        assert bare == [] and copied == [], (bare, copied)

    @pytest.mark.parametrize("module_name", [SCHEDULER, *OWNERS])
    def test_every_qualified_read_names_what_its_owner_defines(self, module_name):
        for alias, name in _qualified_reads(module_name):
            assert name in _top_level_definitions(alias[1:]), f"{alias}.{name}"

    def test_the_scheduler_reads_its_owners_through_qualified_names(self):
        # Not vacuous: the scan finds the reads the scheduler and its
        # collaborators make (the collaborators took many of them, WRITE-5).
        assert len(_qualified_reads(SCHEDULER)) > 50
        assert sum(len(_qualified_reads(name)) for name in [SCHEDULER, *OWNERS]) > 150

    def test_a_patch_on_the_owner_reaches_the_schedulers_own_code(self):
        scheduler = write_batch_scheduler.WriteBatchScheduler.__new__(
            write_batch_scheduler.WriteBatchScheduler)
        scheduler.state = SimpleNamespace(state={"pinned": True})
        scheduler.kind = "apply_code_fixes"
        guard = {"files": {}}
        with mock.patch.object(commit_write_guard, "commit_guard_of",
                               return_value=guard) as fake:
            assert scheduler._commit_guard() is guard
        fake.assert_called_once_with({"pinned": True}, "apply_code_fixes")

    def test_a_patch_on_the_scheduler_reaches_none_of_it(self):
        """The unsupported write the guard below refuses, pinned to its
        documented effect: a shadow in the scheduler that no code reads."""
        scheduler = write_batch_scheduler.WriteBatchScheduler.__new__(
            write_batch_scheduler.WriteBatchScheduler)
        scheduler.state = SimpleNamespace(state={})
        scheduler.kind = "apply_code_fixes"
        with mock.patch.object(write_batch_scheduler, "commit_guard_of",
                               return_value={"never": "read"}, create=True) as fake:
            assert scheduler._commit_guard() is None
        assert fake.call_count == 0
        assert "commit_guard_of" not in vars(write_batch_scheduler)
        assert write_batch_scheduler.commit_guard_of is commit_write_guard.commit_guard_of


# ---------------------------------------------------------------------------
# The layering
# ---------------------------------------------------------------------------


class TestTheLayering:
    @pytest.mark.parametrize("module_name", list(OWNERS))
    def test_a_helper_module_imports_only_the_ones_below_it(self, module_name):
        order = list(OWNERS)
        above = set(order[order.index(module_name) + 1:]) | {SCHEDULER}
        assert not _imported_modules(module_name) & above

    def test_the_scheduler_imports_every_owner(self):
        assert set(OWNERS) <= _imported_modules(SCHEDULER)


# ---------------------------------------------------------------------------
# The guard: no test writes a moved name on the scheduler
# ---------------------------------------------------------------------------


def _guard_message(path, lineno, form, name, owner):
    return (
        f"{path}:{lineno}: {form} writes `{name}` on utils.write_batch_scheduler, but "
        f"{name} lives in utils/{owner}.py and the scheduler only re-exports it for "
        f"reading: the code that uses it never sees this patch. Patch "
        f"utils.{owner}.{name} instead."
    )


def test_no_test_writes_a_moved_name_on_the_scheduler():
    """This file is the one exception: it exercises those writes on purpose."""
    scanner = MovedNameWriteScanner(write_batch_scheduler._MOVED_TO, SCHEDULER)
    tests_dir = Path(__file__).parent
    problems = [
        _guard_message(path.relative_to(SKILL_ROOT), *hit)
        for path in sorted(tests_dir.rglob("*.py"))
        if path != Path(__file__)
        for hit in scanner.scan(path.read_text(encoding="utf-8"), str(path))
    ]
    assert problems == [], "\n".join(problems)


_IMPORTS = "from unittest import mock\nfrom utils import write_batch_scheduler\n"


class TestTheGuardSeesTheScheduler:
    @pytest.mark.parametrize("snippet, name", [
        ('monkeypatch.setattr(write_batch_scheduler, "decomposition_graph", None)',
         "decomposition_graph"),
        ('monkeypatch.setattr("utils.write_batch_scheduler.with_result_contract", None)',
         "with_result_contract"),
        ('from utils import write_batch_scheduler as wbs\n'
         'mock.patch.object(wbs, "compose_stop_answers")', "compose_stop_answers"),
        ('import utils.write_batch_scheduler as scheduler_module\n'
         'scheduler_module.HumanTaskPlan = None', "HumanTaskPlan"),
        ('mock.patch("utils.write_batch_scheduler.commit_guard_of")', "commit_guard_of"),
        ('mock.patch(target="utils.write_batch_scheduler.commit_guard_of")', "commit_guard_of"),
        ('monkeypatch.setattr(target="utils.write_batch_scheduler.commit_guard_of", name=None)',
         "commit_guard_of"),
        ('monkeypatch.setattr(write_batch_scheduler, name="decomposition_graph", value=None)',
         "decomposition_graph"),
        ('mock.patch.object(target=write_batch_scheduler, attribute="compose_stop_answers")',
         "compose_stop_answers"),
    ])
    def test_a_write(self, snippet, name):
        scanner = MovedNameWriteScanner(write_batch_scheduler._MOVED_TO, SCHEDULER)
        hits = scanner.scan(_IMPORTS + snippet)
        assert [hit[2] for hit in hits] == [name]
        owner = write_batch_scheduler._MOVED_TO[name]
        assert f"Patch utils.{owner}.{name} instead" in _guard_message("t.py", *hits[0])

    @pytest.mark.parametrize("snippet", [
        'monkeypatch.setattr(write_batch_scheduler, "dispatch_task", None)',
        'monkeypatch.setattr(write_batch_scheduler, "WRITE_FINGERPRINT_FILENAME", ".x")',
        'monkeypatch.setattr(check_records, "decomposition_graph", None)',
        'mock.patch("utils.check_records.decomposition_graph")',
        'found = write_batch_scheduler.split_accepted_findings([], {})',
    ])
    def test_not_a_write(self, snippet):
        scanner = MovedNameWriteScanner(write_batch_scheduler._MOVED_TO, SCHEDULER)
        assert scanner.scan(_IMPORTS + snippet) == []
