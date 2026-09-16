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
