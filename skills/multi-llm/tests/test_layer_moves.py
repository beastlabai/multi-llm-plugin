"""The names the import-layer moves took out of their old modules
(docs/development/dispatcher.md#import-layers).

A name moved down to the layer its readers need keeps a read-only alias in the
module it left, where a reader (a test, the CLI surface) still spells it there:

* **Reads work.** The old spelling is the owner's object.
* **The old module defines none of them.** Its own code reads the owner.
* **Writes do not.** A patch of the old spelling only shadows it there and
  reaches no code: the code that uses the name reads it in its owner. The
  guard at the end fails any test that writes one, naming the owner to patch.
"""

import ast
import importlib
import sys
from pathlib import Path
from typing import Dict, Tuple

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from tests.moved_name_writes import MovedNameWriteScanner  # noqa: E402

#: Old module -> {old name: (owner module, name there)}.
MOVED: Dict[str, Dict[str, Tuple[str, str]]] = {
    "multi_llm": {
        "_COMMIT_BRIEF_SUFFIX": ("utils.positionals", "COMMIT_BRIEF_SUFFIX"),
        "_RANGE_COUNT_WORD": ("utils.positionals", "RANGE_COUNT_WORD"),
        "_inside": ("utils.positionals", "inside"),
        "_names_a_commit_brief": ("utils.positionals", "names_a_commit_brief"),
        "_relative": ("utils.positionals", "relative_paths"),
        "_revs_flags": ("utils.positionals", "revs_flags"),
        "_root_holding": ("utils.positionals", "root_holding"),
        "_split_positionals": ("utils.positionals", "split_positionals"),
        "_takes_revs": ("utils.positionals", "takes_revs"),
    },
    "utils.progress_tracker": {
        "STATE_DONE": ("utils.status_file", "STATE_DONE"),
        "STATE_FAILED": ("utils.status_file", "STATE_FAILED"),
        "TERMINAL_STATES": ("utils.status_file", "TERMINAL_STATES"),
    },
    "utils.review_orchestrator_base": {
        "get_backoff_delay": ("utils.concurrency", "get_backoff_delay"),
        "get_provider_launch_stagger": ("utils.concurrency", "get_provider_launch_stagger"),
        "plan_launch_limiter": ("utils.concurrency", "plan_launch_limiter"),
        "run_tasks_concurrent": ("utils.concurrency", "run_tasks_concurrent"),
    },
    "utils.tasks_file": {
        "TASKS_FILE_MARKER_PATTERN": ("utils.tasks_marker", "TASKS_FILE_MARKER_PATTERN"),
        "TASKS_FILE_MARKER_RE": ("utils.tasks_marker", "TASKS_FILE_MARKER_RE"),
        "_BACKTICK_RUN_RE": ("utils.tasks_marker", "_BACKTICK_RUN_RE"),
        "_FENCE_OPEN_RE": ("utils.tasks_marker", "_FENCE_OPEN_RE"),
        "_inline_code_ranges": ("utils.tasks_marker", "_inline_code_ranges"),
        "_opens_fence": ("utils.tasks_marker", "_opens_fence"),
        "markdown_code_ranges": ("utils.tasks_marker", "markdown_code_ranges"),
        "tasks_marker": ("utils.tasks_marker", "tasks_marker"),
        "tasks_marker_spans": ("utils.tasks_marker", "tasks_marker_spans"),
    },
    "utils.write_safety": {
        "_REDACTIONS": ("utils.redact", "_REDACTIONS"),
        "_looks_like_credential": ("utils.redact", "_looks_like_credential"),
        "_replace_group": ("utils.redact", "_replace_group"),
        "_sha256_text": ("utils.redact", "_sha256_text"),
        "redact_argv": ("utils.redact", "redact_argv"),
        "redact_text": ("utils.redact", "redact_text"),
        "state_home": ("utils.output_handler", "state_home"),
    },
    "utils.consolidation": {
        "_compute_file_hash": ("utils.output_handler", "_compute_file_hash"),
        "consolidation_is_current": ("utils.output_handler", "consolidation_is_current"),
    },
    "utils.next_phase": {
        "AFTER_IDS_SHOWN": ("utils.manual_tasks", "AFTER_IDS_SHOWN"),
        "GATE_IDS_SHOWN": ("utils.manual_tasks", "GATE_IDS_SHOWN"),
        "GATE_TITLE_CHARS": ("utils.manual_tasks", "GATE_TITLE_CHARS"),
        "MAX_RECORDED_WHY_CHARS": ("utils.manual_tasks", "MAX_RECORDED_WHY_CHARS"),
        "_TASK_ID": ("utils.manual_tasks", "_TASK_ID"),
        "_manual_gate_names": ("utils.manual_tasks", "_manual_gate_names"),
        "after_code_note": ("utils.manual_tasks", "after_code_note"),
        "close_command": ("utils.manual_tasks", "close_command"),
        "gate_resume_closes_the_manual_tasks": ("utils.manual_tasks", "gate_resume_closes_the_manual_tasks"),
        "manual_close_ids": ("utils.manual_tasks", "manual_close_ids"),
        "manual_close_lead": ("utils.manual_tasks", "manual_close_lead"),
        "manual_close_note": ("utils.manual_tasks", "manual_close_note"),
        "manual_gate_decline_answers": ("utils.manual_tasks", "manual_gate_decline_answers"),
        "manual_gate_flags": ("utils.manual_tasks", "manual_gate_flags"),
        "manual_gate_ids": ("utils.manual_tasks", "manual_gate_ids"),
        "manual_gate_why": ("utils.manual_tasks", "manual_gate_why"),
        "manual_note_ids": ("utils.manual_tasks", "manual_note_ids"),
        "manual_note_named": ("utils.manual_tasks", "manual_note_named"),
        "manual_waiting_count": ("utils.manual_tasks", "manual_waiting_count"),
        "waits_on_note": ("utils.manual_tasks", "waits_on_note"),
        "brief_view": ("utils.next_phase_words", "brief_view"),
        "commit_mode_word": ("utils.next_phase_words", "commit_mode_word"),
        "commit_spelled_args": ("utils.next_phase_words", "commit_spelled_args"),
        "commit_view_for": ("utils.next_phase_words", "commit_view_for"),
        "phase_command": ("utils.next_phase_words", "phase_command"),
        "spelled_view": ("utils.next_phase_words", "spelled_view"),
        "ARGS_SLOT": ("utils.routes", "ARGS_SLOT"),
        "RESUME_FLAG": ("utils.routes", "RESUME_FLAG"),
        "SKILL_ROUTE": ("utils.routes", "SKILL_ROUTE"),
        "_PLACEHOLDER_PREFIXES": ("utils.routes", "_PLACEHOLDER_PREFIXES"),
        "_RECORDED_SPELLING": ("utils.routes", "_RECORDED_SPELLING"),
        "_host_capabilities": ("utils.routes", "_host_capabilities"),
        "_install_tail": ("utils.routes", "_install_tail"),
        "_is_skill_install": ("utils.routes", "_is_skill_install"),
        "_path_parts": ("utils.routes", "_path_parts"),
        "_skill_module": ("utils.routes", "_skill_module"),
        "_steps_back_only_out_of_dirs": ("utils.routes", "_steps_back_only_out_of_dirs"),
        "quote_plan_arg": ("utils.routes", "quote_plan_arg"),
        "recorded_plan_spelling": ("utils.routes", "recorded_plan_spelling"),
        "resolve_route": ("utils.routes", "resolve_route"),
    },
}

#: Old modules that moved whole: every name of the owner reads through them
#: (a module ``__getattr__``), and only the CLI entry stays defined there.
MOVED_WHOLE: Dict[str, str] = {
    "finalize_tracking": "utils.tracking",
}


def _module(name):
    return importlib.import_module(name)


def _source_path(module_name):
    return SKILL_ROOT / Path(*module_name.split(".")).with_suffix(".py")


def _top_level_definitions(module_name):
    names = set()
    for node in ast.parse(_source_path(module_name).read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _whole_names(owner):
    return {name for name in vars(_module(owner)) if not name.startswith("__")}


def _moved_table(old):
    """{old name: "owner.name"} for every name *old* re-exports."""
    if old in MOVED_WHOLE:
        owner = MOVED_WHOLE[old]
        return {name: f"{owner}.{name}" for name in _whole_names(owner)}
    return {name: f"{owner}.{there}" for name, (owner, there) in MOVED[old].items()}


_ALIASES = [(old, name, owner, there)
            for old, names in MOVED.items() for name, (owner, there) in names.items()]


@pytest.mark.parametrize("old, name, owner, there", _ALIASES,
                         ids=[f"{a[0]}.{a[1]}" for a in _ALIASES])
def test_a_moved_name_reads_as_the_owners_object(old, name, owner, there):
    assert getattr(_module(old), name) is getattr(_module(owner), there)


@pytest.mark.parametrize("old", sorted(MOVED))
def test_the_old_module_defines_none_of_them(old):
    assert not _top_level_definitions(old) & set(MOVED[old])


@pytest.mark.parametrize("old, owner", sorted(MOVED_WHOLE.items()))
def test_a_module_moved_whole_reads_through_to_its_owner(old, owner):
    module, target = _module(old), _module(owner)
    for name in _whole_names(owner):
        assert getattr(module, name) is getattr(target, name), name
    assert set(_whole_names(owner)) <= set(dir(module))
    assert _top_level_definitions(old) == {"__getattr__", "__dir__"}
    with pytest.raises(AttributeError, match="has no attribute 'not_a_name'"):
        module.not_a_name  # noqa: B018


# ---------------------------------------------------------------------------
# The guard: no test writes a moved name on the module it left
# ---------------------------------------------------------------------------


def _guard_message(old, path, lineno, form, name, owner):
    return (
        f"{path}:{lineno}: {form} writes `{name}` on {old}, which only "
        f"re-exports it for reading: the code that uses it reads {owner}. "
        f"Patch {owner} instead."
    )


@pytest.mark.parametrize("old", sorted({*MOVED, *MOVED_WHOLE}))
def test_no_test_writes_a_moved_name_on_its_old_module(old):
    """This file is the one exception: it exercises those writes on purpose."""
    scanner = MovedNameWriteScanner(_moved_table(old), old.rpartition(".")[2])
    tests_dir = Path(__file__).parent
    problems = [
        _guard_message(old, path.relative_to(SKILL_ROOT), *hit)
        for path in sorted(tests_dir.rglob("*.py"))
        if path != Path(__file__)
        for hit in scanner.scan(path.read_text(encoding="utf-8"), str(path))
    ]
    assert problems == [], "\n".join(problems)


@pytest.mark.parametrize("snippet, old, name", [
    ('import multi_llm\nmonkeypatch.setattr(multi_llm, "_split_positionals", None)',
     "multi_llm", "_split_positionals"),
    ('from unittest import mock\nmock.patch("multi_llm._revs_flags")',
     "multi_llm", "_revs_flags"),
    ('import finalize_tracking\nmonkeypatch.setattr(finalize_tracking, "main", None)',
     "finalize_tracking", "main"),
    ('from unittest import mock\nmock.patch("finalize_tracking.get_all_modified_files")',
     "finalize_tracking", "get_all_modified_files"),
])
def test_the_guard_sees_a_write(snippet, old, name):
    hits = MovedNameWriteScanner(_moved_table(old), old).scan(snippet)
    assert [hit[2] for hit in hits] == [name]


@pytest.mark.parametrize("snippet, old", [
    ('from utils import positionals\n'
     'monkeypatch.setattr(positionals, "split_positionals", None)', "multi_llm"),
    ('import multi_llm\nfound = multi_llm._split_positionals(["a"])', "multi_llm"),
    ('from utils import tracking\nmonkeypatch.setattr(tracking, "main", None)',
     "finalize_tracking"),
])
def test_the_guard_leaves_reads_and_owner_patches_alone(snippet, old):
    assert MovedNameWriteScanner(_moved_table(old), old).scan(snippet) == []
