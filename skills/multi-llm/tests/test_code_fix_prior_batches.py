"""An apply-code-fixes batch's prior-batch context (live run 9, #7).

Round 20 stopped the batch prompt from quoting implement's record twice
(`quoted_in`), but two kinds of noise were left under `## Changes Applied in
Prior Batches`:

* Claude Code racing 08: implement's entries past the record's five-entry
  budget (T002, T006, T007, T008) were listed again as "Batch 15-18", labels
  that name no batch of anything;
* Codex racing 08: a phase's first batch has no entries of its own, so the
  block fell back to the whole log and listed the plan-edit batches of
  apply-suggestions and apply-task-suggestions in a code-fix prompt.

Now (fix round 22, both legs through `bind_prior_changes`): the record shows
implement's entries by task id within its budget and names what it left out
("and 4 more tasks: T002, T006, T007, T008"); the block lists only the earlier
batches of this apply-code-fixes run — never another phase's, never a
finished earlier run's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from utils import prior_changes, write_prompt  # noqa: E402
from utils.code_fix_batcher import CodeFixBatch, format_fix_batch_prompt  # noqa: E402
from utils.subagent_tasks import PRIOR_CHANGES_TOKEN  # noqa: E402
from utils.write_batch_scheduler import WriteBatchScheduler, bind_prior_changes  # noqa: E402
from utils.write_phase_bridge import (  # noqa: E402
    materialize_write_tasks,
    write_task_file_for_phase,
)

HEADING = "## Changes Applied in Prior Batches"

#: Claude Code racing 08's implement record: five entries name index.html.
IMPLEMENT = [
    ("T001", "Created index.html: importmap, renderer, ground plane."),
    ("T002", "Added closed 10-point CatmullRom track and road mesh."),
    ("T003", "Added start/finish line and kerbs to index.html."),
    ("T004", "Added createCarMesh and the car state to index.html."),
    ("T005", "Added keyboard input and the chase camera in index.html."),
    ("T006", "Added computeOpponentInput in the Opponents section."),
    ("T007", "Added per-car race state and getRanking."),
    ("T008", "Added countdown, HUD and results overlay."),
    ("T009", "Reviewed loop and failure paths in index.html."),
]
#: Codex racing 08: plan-edit batches of the two planning apply phases.
PLAN_EDITS = [
    ("section:goal#f6ff4fc2", "apply-suggestions",
     "Added Chrome and Firefox file:// verification."),
    ("section:t002#7f2c2397", "apply-task-suggestions",
     "Clarified T002 track validation."),
]


def _line(entry_id: str, phase: str, summary: str, when: str) -> str:
    return json.dumps({"id": entry_id, "phase": phase, "summary": summary,
                       "timestamp": when})


def _log(out: Path, lines) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / prior_changes.FILENAME).open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def _workflow_log(out: Path) -> None:
    """The plan-edit batches, then implement's nine tasks, as run 9 logged them."""
    _log(out, [
        _line(entry_id, phase, summary, "2026-10-07T13:40:00")
        for entry_id, phase, summary in PLAN_EDITS
    ] + [
        _line(task_id, "implement", summary, f"2026-10-07T13:5{n}:00")
        for n, (task_id, summary) in enumerate(IMPLEMENT)
    ])


def _fix_prompt(out: Path) -> str:
    batch = CodeFixBatch(
        fixes=[{"title": "Initialize camera aspect", "file": "index.html",
                "description": "d", "type": "bug", "importance": "MEDIUM"}],
        file_key="index.html", batch_type="bug",
    )
    return format_fix_batch_prompt(batch, "plans/racing-game.md", output_dir=str(out))


def _record(prompt: str) -> str:
    tail = prompt.split("## Implementation record", 1)[1]
    return tail.split("If a finding above contradicts", 1)[0]


def _block(prompt: str) -> str:
    tail = prompt.split(HEADING, 1)[1]
    return tail.split("\n---", 1)[0].split("\n## ", 1)[0]


def _bind(out: Path, prompt: str, task_id: str = "apply-fix-batch-0") -> str:
    return bind_prior_changes(
        {"task_id": task_id, "prompt": prompt},
        output_dir=str(out), phase="apply-code-fixes",
    )


def test_the_record_names_the_tasks_its_budget_left_out(tmp_path):
    _workflow_log(tmp_path)
    record = _record(_fix_prompt(tmp_path))
    for task_id in ("T001", "T003", "T004", "T005", "T009"):
        assert f"- ({task_id}): " in record, record
    assert "- and 4 more tasks: T002, T006, T007, T008" in record, record


def test_a_record_within_its_budget_says_nothing_more(tmp_path):
    _log(tmp_path, [_line("T001", "implement", "Created index.html.", "2026-10-07T13:50:00")])
    record = _record(_fix_prompt(tmp_path))
    assert "- (T001): Created index.html." in record
    assert "more task" not in record


def test_the_first_batch_lists_nothing_from_another_phase(tmp_path):
    _workflow_log(tmp_path)
    bound = _bind(tmp_path, _fix_prompt(tmp_path))
    block = _block(bound)
    assert prior_changes.DEFAULT_TEXT in block, block
    assert "**Batch" not in block, block
    assert "other phases" not in block and "already quoted" not in block, block
    for _id, _phase, summary in PLAN_EDITS:
        assert summary not in bound
    for task_id, summary in IMPLEMENT:
        # Shown once, in the record, or named there by id alone.
        assert bound.count(summary) <= 1, summary
        assert task_id in bound


def test_a_later_batch_lists_the_earlier_batches_of_this_run_only(tmp_path):
    _workflow_log(tmp_path)
    _log(tmp_path, [_line("index.html#d7631f8e", "apply-fixes",
                          "Initialize camera aspect and renderer size.",
                          "2026-10-07T14:11:33")])
    block = _block(_bind(tmp_path, _fix_prompt(tmp_path), "apply-fix-batch-1"))
    assert block.strip().splitlines()[-1] == (
        "1. **Batch 1** (index.html#d7631f8e): Initialize camera aspect and "
        "renderer size."
    ), block
    assert "other phases" not in block and "already quoted" not in block


def _state(out: Path, completed_at: str) -> None:
    (out / "state.json").write_text(json.dumps({
        "phases_completed": {"apply-code-fixes": completed_at},
    }), encoding="utf-8")


def test_an_earlier_apply_code_fixes_run_is_not_this_runs(tmp_path):
    _workflow_log(tmp_path)
    _log(tmp_path, [
        _line("index.html#0ld0ld00", "apply-fixes", "Cycle 1: clamped the speed.",
              "2026-10-07T14:00:00"),
        _line("index.html#d7631f8e", "apply-fixes", "Cycle 2: camera aspect.",
              "2026-10-07T15:00:00"),
    ])
    _state(tmp_path, "2026-10-07T14:30:00")
    block = _block(_bind(tmp_path, _fix_prompt(tmp_path), "apply-fix-batch-1"))
    assert "Cycle 1" not in block, block
    assert "1. **Batch 1** (index.html#d7631f8e): Cycle 2: camera aspect." in block


def test_both_legs_bind_the_same_section(tmp_path):
    plan = tmp_path / "plans" / "racing-game.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Racing\n", encoding="utf-8")
    out = tmp_path / "plans" / "racing-game"
    phase_dir = out / "apply-fixes"
    phase_dir.mkdir(parents=True)
    _workflow_log(out)
    _log(out, [
        _line("index.html#0ld0ld00", "apply-fixes", "Cycle 1: clamped the speed.",
              "2026-10-07T14:00:00"),
        _line("index.html#d7631f8e", "apply-fixes", "Cycle 2: camera aspect.",
              "2026-10-07T15:00:00"),
    ])
    _state(out, "2026-10-07T14:30:00")
    prompt = _fix_prompt(out)
    assert PRIOR_CHANGES_TOKEN in prompt
    batches = [
        {"file_key": "index.html", "prompt": prompt,
         "fixes": [{"group_id": f"g{n}", "title": f"Fix {n}"}]}
        for n in range(2)
    ]
    tasks = materialize_write_tasks(
        kind="apply_code_fixes", batches=batches, phase_dir=str(phase_dir),
        plan_file=str(plan),
    )
    write_task_file_for_phase(
        kind="apply_code_fixes", tasks=tasks, phase_dir=str(phase_dir),
        plan_file=str(plan),
    )
    native, _task, info = write_prompt.render_write_prompt(phase_dir, "apply-fix-batch-1")
    scheduler = WriteBatchScheduler.__new__(WriteBatchScheduler)
    scheduler.kind = "apply_code_fixes"
    scheduler.output_dir = str(out)
    scheduler.phase = "apply-code-fixes"
    scheduler.state = json.loads((out / "state.json").read_text())
    scheduler.tasks = tasks
    scheduler.plan_file = str(plan)
    executor = scheduler._with_result_contract(
        tasks[1], scheduler.dispatched_prompt(tasks[1]))
    assert native == executor
    block = _block(native)
    assert "Cycle 2: camera aspect." in block and "Cycle 1" not in block
    for _id, _phase, summary in PLAN_EDITS:
        assert summary not in native
    assert info["context_entries"] == 1


def test_an_earlier_runs_entry_is_not_this_batchs_for_the_order_warning(tmp_path):
    """`util write-prompt`'s `[WRITE_PROMPT_ORDER]` counts as logged only
    what the render binds: an entry of the same id an earlier, finished run
    recorded is not this run's batch 0."""
    plan = tmp_path / "plans" / "racing-game.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Racing\n", encoding="utf-8")
    out = tmp_path / "plans" / "racing-game"
    phase_dir = out / "apply-fixes"
    phase_dir.mkdir(parents=True)
    batches = [
        {"file_key": "index.html", "prompt": f"Fix {n}.\n\n{PRIOR_CHANGES_TOKEN}\n",
         "fixes": [{"group_id": f"g{n}", "title": f"Fix {n}"}]}
        for n in range(2)
    ]
    tasks = materialize_write_tasks(
        kind="apply_code_fixes", batches=batches, phase_dir=str(phase_dir),
        plan_file=str(plan),
    )
    write_task_file_for_phase(
        kind="apply_code_fixes", tasks=tasks, phase_dir=str(phase_dir),
        plan_file=str(plan),
    )
    _log(out, [_line(tasks[0]["prior_changes_id"], "apply-fixes",
                     "Cycle 1: the same fix, an earlier run.", "2026-10-07T14:00:00")])
    (out / "state.json").write_text(json.dumps({
        "phases_completed": {"apply-code-fixes": "2026-10-07T14:30:00"},
        "task_status": {"apply-fix-batch-0": "completed"},
    }), encoding="utf-8")
    lines = write_prompt.ordering_warnings(phase_dir, "apply-fix-batch-1")
    assert len(lines) == 1 and "apply-fix-batch-0 is recorded completed" in lines[0], lines
    native, _task, _info = write_prompt.render_write_prompt(phase_dir, "apply-fix-batch-1")
    assert "Cycle 1" not in native
