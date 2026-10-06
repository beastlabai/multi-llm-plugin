"""Regression coverage for containment sweeps over overlapping write tasks."""

import pytest

from tests.test_write_executor import env  # noqa: F401


@pytest.mark.parametrize("first_finished", ["A", "B"])
def test_completed_parallel_sibling_is_not_adopted_for_rollback(env, first_finished):
    tasks = {
        label: env.apply_task(
            index, task_id=label,
            declared_files=[str(env.root / "src" / f"{label.lower()}.py")],
        )
        for index, label in enumerate(("A", "B"))
    }
    scheduler = env.scheduler(list(tasks.values()))
    snapshots = {label: scheduler._snapshot(task) for label, task in tasks.items()}
    for label in tasks:
        scheduler._begin_sweep(label, [f"src/{label.lower()}.py"])
    for label in tasks:
        (env.root / "src" / f"{label.lower()}.py").write_text(f"{label} = 2\n")

    second_finished = "B" if first_finished == "A" else "A"
    for label in (first_finished, second_finished):
        observed = scheduler._undeclared_writes(label, snapshots[label])
        assert observed.paths == []
        assert observed.restorable == []

    # A later, non-overlapping task must still observe a write to either file.
    later = env.apply_task(2, task_id="later")
    snapshot = scheduler._snapshot(later)
    scheduler._begin_sweep("later", ["plans/p.md"])
    (env.root / "src" / "a.py").write_text("unexpected = 3\n")
    assert scheduler._undeclared_writes("later", snapshot).paths == ["src/a.py"]


# ---------------------------------------------------------------------------
# Fix round 15 (review of round 14): a code task whose declared files did not
# change is applied on its stray writes alone ("the bytes decide"), so a
# co-scheduled sibling's stray file — seen by both sweeps — turned a task that
# did nothing into "applied". A path a sibling running at the same time also
# saw move is not counted as this task's change.
# ---------------------------------------------------------------------------

def _code_tasks(env):
    from utils import write_batch_scheduler as wbs

    return {
        label: env.apply_task(
            index, kind="implement", task_id=label, expected_ids=[label],
            declared_files=[str(env.root / "src" / f"{label.lower()}.py")],
            prompt=f"Do {label}\n\n## Changes Applied in Prior Batches\n\n"
                   f"{wbs.PRIOR_CHANGES_TOKEN}\n",
        )
        for index, label in enumerate(("A", "B"))
    }


def _classify(scheduler, task, snapshot):
    from utils import subagent_executor as executor

    label = task["task_id"]
    profile = executor.profile_for_kind("implement")
    outcome = executor.TaskOutcome(
        task_id=label, kind="implement", runner="native",
        status=executor.STATUS_OK, profile=profile,
    )
    scheduler._classify(
        task, outcome, snapshot,
        {"status": "applied", "ids": [label],
         "files_touched": [f"src/{label.lower()}.py"], "summary": "did it"},
        runner="native", profile=profile, attempt=1, exit_code=0, duration=0.0,
    )
    return outcome


@pytest.mark.parametrize("first", ["A", "B"])
def test_a_stray_a_running_sibling_also_saw_does_not_make_a_task_applied(env, first):
    tasks = _code_tasks(env)
    scheduler = env.scheduler(list(tasks.values()), kind="implement", finalize=False)
    snapshots = {label: scheduler._snapshot(task) for label, task in tasks.items()}
    for label in tasks:
        scheduler._begin_sweep(label, [f"src/{label.lower()}.py"])
    # While both run: B edits its own file and leaves a stray; A edits nothing.
    (env.root / "notes.md").write_text("stray\n", encoding="utf-8")
    (env.root / "src" / "b.py").write_text("b = 2\n", encoding="utf-8")

    order = [first, "B" if first == "A" else "A"]
    outcomes = {label: _classify(scheduler, tasks[label], snapshots[label])
                for label in order}

    a = outcomes["A"]
    assert not a.ok, a.reason
    assert a.applied_without_changes
    assert "notes.md" in a.reason and "B" in a.reason, a.reason
    assert outcomes["B"].ok, outcomes["B"].reason


def test_a_stray_written_before_the_sibling_started_is_still_the_tasks_own(env):
    tasks = _code_tasks(env)
    scheduler = env.scheduler(list(tasks.values()), kind="implement", finalize=False)
    snapshots = {label: scheduler._snapshot(task) for label, task in tasks.items()}
    scheduler._begin_sweep("A", ["src/a.py"])
    (env.root / "src" / "a_helper.py").write_text("x = 1\n", encoding="utf-8")
    scheduler._begin_sweep("B", ["src/b.py"])
    (env.root / "src" / "b.py").write_text("b = 2\n", encoding="utf-8")

    a = _classify(scheduler, tasks["A"], snapshots["A"])
    assert a.ok, a.reason
    assert a.reason.startswith("APPLIED outside the declared files")
    assert "src/a_helper.py" in a.files_touched
    assert _classify(scheduler, tasks["B"], snapshots["B"]).ok
