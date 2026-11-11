"""``WriteBatchScheduler`` is built from one ``SchedulerOptions`` (WRITE-5).

A frozen keyword-only dataclass whose fields are the constructor's options:
the orchestrators build one and pass it, a test may pass the same fields as
keywords, which builds the same record, and never both at once
(docs/development/write-scheduler.md#where-the-schedulers-helpers-live).
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_DIR))

from utils.write_batch_scheduler import SchedulerOptions, WriteBatchScheduler  # noqa: E402
from utils.write_context import NullState  # noqa: E402


def _fields(tmp_path, **over):
    fields = dict(tasks=[], phase_dir=str(tmp_path), kind="implement",
                  output_dir=str(tmp_path), repo_root=str(tmp_path),
                  log=lambda line: None)
    fields.update(over)
    return fields


class TestTheOptions:
    def test_they_are_frozen_and_keyword_only(self, tmp_path):
        options = SchedulerOptions(**_fields(tmp_path))
        with pytest.raises(dataclasses.FrozenInstanceError):
            options.kind = "apply_suggestions"
        with pytest.raises(TypeError):
            SchedulerOptions([], str(tmp_path), "implement", str(tmp_path))

    def test_the_four_without_a_default_are_required(self):
        required = [f.name for f in dataclasses.fields(SchedulerOptions)
                    if f.default is dataclasses.MISSING]
        assert required == ["tasks", "phase_dir", "kind", "output_dir"]
        with pytest.raises(TypeError):
            SchedulerOptions(tasks=[], phase_dir="p", kind="implement")


class TestBuildingTheScheduler:
    def test_options_and_keywords_build_the_same_scheduler(self, tmp_path):
        fields = _fields(tmp_path, resume=False, partial_skips=["T001"],
                         user_commands={"resume": "go"}, host_max=2)
        by_options = WriteBatchScheduler(SchedulerOptions(**fields))
        by_keywords = WriteBatchScheduler(**fields)
        assert by_options.options == by_keywords.options == SchedulerOptions(**fields)
        for name in ("kind", "phase", "phase_dir", "output_dir", "repo_root", "resume",
                     "partial_skips", "user_commands", "host_max", "invoker"):
            assert getattr(by_options, name) == getattr(by_keywords, name), name
        assert isinstance(by_options.state, NullState)

    def test_the_scheduler_keeps_the_options_it_was_given(self, tmp_path):
        options = SchedulerOptions(**_fields(tmp_path))
        assert WriteBatchScheduler(options).options is options

    def test_both_at_once_another_object_or_an_unknown_field_is_refused(self, tmp_path):
        options = SchedulerOptions(**_fields(tmp_path))
        with pytest.raises(TypeError, match="not both"):
            WriteBatchScheduler(options, kind="implement")
        with pytest.raises(TypeError, match="not both"):
            WriteBatchScheduler(_fields(tmp_path))
        with pytest.raises(TypeError):
            WriteBatchScheduler(**_fields(tmp_path), not_an_option=1)

    def test_an_unknown_kind_is_refused_as_before(self, tmp_path):
        from utils.subagent_executor import ExecutorError

        with pytest.raises(ExecutorError, match="is not a write kind"):
            WriteBatchScheduler(SchedulerOptions(**_fields(tmp_path, kind="review")))


def _called_name(func):
    """``WriteBatchScheduler`` for ``WriteBatchScheduler(...)`` and for
    ``write_batch_scheduler.WriteBatchScheduler(...)``, the spelling that reads
    the class through its module at the call, where a test patches it."""
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)


@pytest.mark.parametrize("path", ["implement_orchestrator.py", "utils/apply_orchestrator_base.py"])
def test_each_orchestrator_passes_one_options_record(path):
    calls = [node for node in ast.walk(ast.parse((SKILL_DIR / path).read_text()))
             if isinstance(node, ast.Call) and _called_name(node.func) == "WriteBatchScheduler"]
    assert len(calls) == 1
    (call,) = calls
    assert not call.keywords and len(call.args) == 1
    assert getattr(call.args[0].func, "id", None) == "SchedulerOptions"
