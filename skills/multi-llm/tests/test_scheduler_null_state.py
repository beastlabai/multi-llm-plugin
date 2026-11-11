"""A write run without a state holds a ``NullState`` (``utils/write_context.py``).

``SchedulerOptions.state`` left ``None`` is a unit test driving the loop: a real
phase always has a ``StateManager`` (WRITE-11). The run holds a null object in
its place, so the scheduler and its collaborators call it rather than ask
whether it is there. These pin that it writes nothing and reads as a missing
state read before: its methods are the ``StateManager``'s, it has no metric
methods, every reader it is handed finds what ``None`` found, a reader the
native leg shares is still handed ``None`` (and reads ``state.json`` on
disk), and the two places where a missing state is not an empty one still
tell them apart (docs/development/write-scheduler.md#where-the-schedulers-helpers-live).
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.scheduler_support import make_scheduler  # noqa: E402
from tests.support.declined_world import _codex_08_respond, _codex_08_tasks  # noqa: E402
from tests.test_write_executor import env  # noqa: E402,F401
from utils import batch_prompt as batch_prompt_mod  # noqa: E402
from utils import batch_prompt_builder  # noqa: E402
from utils import check_records  # noqa: E402
from utils import human_task_plan  # noqa: E402
from utils import known_checks  # noqa: E402
from utils import stop_answers  # noqa: E402
from utils import write_batch_scheduler as wbs  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.subagent_tasks import implement_output_name  # noqa: E402
from utils.write_context import NullState, has_state, manager_or_none  # noqa: E402

#: An implement decomposition: a code task, a check after it, a code task
#: after the check, and a manual task.
TASKS = [
    {"task_id": "T001", "declared_files": ["src/a.py"], "depends_on": []},
    {"task_id": "T002", "declared_files": [], "depends_on": ["T001"],
     "task_title": "Run the tests"},
    {"task_id": "T003", "declared_files": ["src/b.py"], "depends_on": ["T002"]},
    {"task_id": "T004", "executor_role": "human", "is_human": True,
     "depends_on": ["T001"]},
]

#: The ``StateManager`` methods a write run calls that only a state which
#: keeps metrics has (``EtaLines`` asks for them by name).
METRIC_METHODS = ("record_metric", "set_metrics_total", "start_metrics_phase",
                  "set_metrics_left_out", "set_metrics_set_aside",
                  "finish_metrics_phase")


def _methods(cls):
    return {name for name, value in vars(cls).items()
            if callable(value) and not name.startswith("_")}


class TestTheNullObject:
    def test_every_method_is_the_state_managers_with_its_signature(self):
        for name in sorted(_methods(NullState)):
            assert name in vars(StateManager), name
            ours = list(inspect.signature(getattr(NullState, name)).parameters.values())
            theirs = list(inspect.signature(getattr(StateManager, name)).parameters.values())
            assert [(p.name, p.kind, p.default) for p in ours] == [
                (p.name, p.kind, p.default) for p in theirs], name

    def test_it_keeps_no_metrics(self):
        for name in METRIC_METHODS:
            assert hasattr(StateManager, name), name
            assert not hasattr(NullState(), name), name

    def test_writes_keep_nothing_and_reads_are_the_defaults(self):
        state = NullState()
        state.set("task_titles", {"T001": "x"})
        state.update_task_status("T001", "completed", reason="done")
        state.record_dependency_override("T003", ["T002"])
        state.mark_phase_started("implement")
        state.mark_phase_completed("implement")
        state.refresh_plan_hash()
        state.save()
        state.reload()
        assert vars(state) == {}
        assert state.state is None and state.state_file is None
        assert state.get("task_titles") is None
        assert state.get("task_titles", "fallback") == "fallback"
        assert state.get_task_status("T001") is None
        assert state.get_all_task_statuses() == {}
        assert not state.is_phase_started("implement")
        assert not state.is_phase_completed("implement")
        assert not state.completion_predates("apply-code-fixes", "code-review")

    def test_has_state_and_manager_or_none(self):
        real = SimpleNamespace(state={})
        assert not has_state(NullState()) and not has_state(None)
        assert has_state(real)
        assert manager_or_none(NullState()) is None and manager_or_none(None) is None
        assert manager_or_none(real) is real


class TestARunWithoutAState:
    def test_the_scheduler_and_its_collaborators_hold_one_null_state(self, tmp_path):
        scheduler = make_scheduler(tmp_path, kind="implement")
        assert isinstance(scheduler.state, NullState)
        assert scheduler._ctx.state is scheduler.state
        given = SimpleNamespace(state={}, state_file=None)
        assert make_scheduler(tmp_path, kind="implement", state=given).state is given

    def test_a_whole_run_writes_no_state_json(self, env):
        calls = []
        result = env.scheduler(
            _codex_08_tasks(env), respond=_codex_08_respond(env, calls),
        ).run()
        assert calls == ["apply-batch-0"] and result.exit_code == 3
        assert not list(env.root.rglob("state.json"))


@pytest.mark.parametrize("read", [
    pytest.param(lambda s: check_records.rerunnable_checks(s), id="rerunnable_checks"),
    pytest.param(lambda s: check_records.decomposition_graph(TASKS, s), id="decomposition_graph"),
    pytest.param(lambda s: check_records.accepted_findings_for(TASKS, TASKS[2], s),
                 id="accepted_findings_for"),
    pytest.param(lambda s: check_records.repeated_findings_for(TASKS, TASKS[2], s),
                 id="repeated_findings_for"),
    pytest.param(lambda s: check_records.accepted_upstream_checks(TASKS, "T003", s),
                 id="accepted_upstream_checks"),
    pytest.param(lambda s: check_records.reported_checks_on_record(s),
                 id="reported_checks_on_record"),
    pytest.param(lambda s: check_records.accepted_checks_on_record(s),
                 id="accepted_checks_on_record"),
    pytest.param(lambda s: human_task_plan.answered_done_on_record(s, exclude=["T001"]),
                 id="answered_done_on_record"),
    pytest.param(lambda s: stop_answers.implement_tasks_done(s), id="implement_tasks_done"),
])
def test_every_reader_it_is_handed_finds_what_none_found(read):
    """The readers the scheduler and its collaborators hand the run's state to."""
    assert read(NullState()) == read(None)


def test_recording_a_human_plan_into_it_writes_nothing():
    plan = human_task_plan.resolve_human_task_plan(
        TASKS, strategy="skip-continue", decisions={}, check_findings_ids=(),
        graph=TASKS,
    )
    before = (dict(plan.skipped), list(plan.completed), dict(plan.overrides))
    state = NullState()
    human_task_plan.persist_human_task_plan(state, plan)
    assert vars(state) == {}
    assert (dict(plan.skipped), list(plan.completed), dict(plan.overrides)) == before


class TestASharedReaderIsHandedNone:
    """``util write-prompt``'s readers read ``state.json`` on disk for ``None``;
    a run without a state still hands them ``None``, never the null object."""

    def test_the_prompt_and_the_known_checks(self, tmp_path, monkeypatch):
        scheduler = make_scheduler(tmp_path, kind="apply_code_fixes")
        seen = {}

        def bind(task, **kwargs):
            seen["bind"] = kwargs["state"]
            return "prompt"

        def commands(output_dir, **kwargs):
            seen["commands"] = kwargs["state"]
            return [("make test", "Makefile")]

        def read_state(output_dir):
            seen.setdefault("read_state", []).append(str(output_dir))
            return {}

        monkeypatch.setattr(batch_prompt_mod, "bind_prior_changes", bind)
        monkeypatch.setattr(known_checks, "known_check_commands", commands)
        monkeypatch.setattr(known_checks, "read_state", read_state)
        assert scheduler._prompts.dispatched_prompt({"task_id": "fix-batch-0"}) == "prompt"
        scheduler._prompts.known_checks_section({"task_id": "fix-batch-0"})
        scheduler._known_checks_report()
        assert seen["bind"] is None and seen["commands"] is None
        assert seen["read_state"] == [str(tmp_path), str(tmp_path)]
        assert batch_prompt_builder._known_checks is known_checks


class TestWhereAMissingStateIsNotAnEmptyOne:
    def _check_failed_unchanged(self, tmp_path, state):
        scheduler = make_scheduler(tmp_path, kind="implement", state=state)
        task = {"task_id": "T002", "kind": "implement",
                "output_name": implement_output_name("T002"),
                "expected_ids": [], "declared_files": []}
        (tmp_path / implement_output_name("T002")).write_text(json.dumps({
            "status": "failed", "summary": "tests fail",
            "verdict": {"ok": False, "verification_only": True,
                        "tree_fingerprint": "tree-1"},
        }), encoding="utf-8")
        scheduler._checks.tree_fingerprint = lambda: "tree-1"
        result = wbs.SchedulerResult(kind="implement", phase="implement")
        return scheduler._checks.unchanged_since_failure(task, result)

    def test_a_failed_check_with_no_state_goes_by_its_result_file(self, tmp_path):
        refused = self._check_failed_unchanged(tmp_path, None)
        assert refused is not None and refused.not_rerun
        assert refused.reason.startswith("NOT RE-RUN: nothing in the repository changed")

    def test_with_a_state_it_goes_by_the_recorded_status(self, tmp_path):
        empty = SimpleNamespace(state={}, get_task_status=lambda task_id: None)
        assert self._check_failed_unchanged(tmp_path, empty) is None

    def test_no_state_has_no_manual_gate(self, tmp_path):
        gates = {"gated": ["T003"], "waiting": ["T004"]}

        def gate(state):
            scheduler = make_scheduler(tmp_path, kind="implement", state=state)
            scheduler._stop.manual_gates = lambda plan: gates
            return scheduler._stop.manual_gate(
                wbs.SchedulerResult(kind="implement", phase="implement"))

        empty = SimpleNamespace(state={}, is_phase_completed=lambda phase: False)
        assert gate(None) is None
        assert gate(empty) == gates
