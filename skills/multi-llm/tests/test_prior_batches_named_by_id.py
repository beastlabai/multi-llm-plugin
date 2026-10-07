"""A prompt names each earlier batch by its id, once (live run 10, #6).

Codex and OpenCode racing 08: the apply-code-fixes prompts listed this run's
earlier batches under ``## Changes Applied in Prior Batches`` as "Batch 1",
"Batch 2", "Batch 3" while those batches are ``apply-fix-batch-0/1/2`` — the
id summary.md, ``status`` and ``--human-task <batch>=skip`` use, so "Batch 1"
read as the second batch. And the ``## Implementation record`` line of a
task whose subagent opened its summary with its own id read
"- (T002): T002: 8-point closed CatmullRom track…".

Now each prior-changes entry carries the id of the batch that wrote it
(``task_id``, recorded by both legs' append: the executor passes it, the
native ``util prior-changes append`` reads it off the phase's
``write_tasks.json``), every list of earlier batches names that id (with the
entry id beside it when the two differ), and no line prints an id twice.
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
from utils.write_batch_scheduler import WriteBatchScheduler  # noqa: E402
from utils.write_phase_bridge import (  # noqa: E402
    WRITE_TASK_FILENAME,
    materialize_write_tasks,
    write_task_file_for_phase,
)

from tests.test_write_executor import FakeState, env  # noqa: E402,F401

HEADING = "## Changes Applied in Prior Batches"


def _block(prompt: str) -> str:
    """The section's entries, below its "(Review these …)" line."""
    tail = prompt.split(HEADING, 1)[1]
    body = tail.split("\n---", 1)[0].split("\n## ", 1)[0]
    return "\n".join(
        line for line in body.strip().splitlines()[1:] if line.strip()
    )


def _fix_prompt(out: Path) -> str:
    batch = CodeFixBatch(
        fixes=[{"title": "Clamp the timer", "file": "index.html",
                "description": "d", "type": "bug", "importance": "LOW"}],
        file_key="index.html", batch_type="bug",
    )
    return format_fix_batch_prompt(batch, "plans/racing-game.md", output_dir=str(out))


class _Racing08:
    """Codex racing 08's apply-code-fixes phase: one batch per fix on index.html."""

    def __init__(self, tmp_path: Path, batches: int = 4) -> None:
        self.plan = tmp_path / "plans" / "racing-game.md"
        self.plan.parent.mkdir(parents=True)
        self.plan.write_text("# Racing\n", encoding="utf-8")
        self.out = tmp_path / "plans" / "racing-game"
        self.phase_dir = self.out / "apply-fixes"
        self.phase_dir.mkdir(parents=True)
        prompt = _fix_prompt(self.out)
        assert PRIOR_CHANGES_TOKEN in prompt
        self.tasks = materialize_write_tasks(
            kind="apply_code_fixes",
            batches=[
                {"file_key": "index.html", "prompt": prompt,
                 "fixes": [{"group_id": f"g{n}", "title": f"Fix {n}"}]}
                for n in range(batches)
            ],
            phase_dir=str(self.phase_dir), plan_file=str(self.plan),
        )
        write_task_file_for_phase(
            kind="apply_code_fixes", tasks=self.tasks,
            phase_dir=str(self.phase_dir), plan_file=str(self.plan),
        )

    def native_append(self, index: int, summary: str) -> None:
        """What the native host runs (instructions/apply-code-fixes.md step 7)."""
        assert prior_changes.main([
            "append", "--output-dir", str(self.out),
            "--id", self.tasks[index]["prior_changes_id"],
            "--phase", "apply-fixes", "--summary", summary,
        ]) == 0

    def native_prompt(self, index: int) -> str:
        text, _task, _info = write_prompt.render_write_prompt(
            self.phase_dir, f"apply-fix-batch-{index}")
        return text

    def executor_prompt(self, index: int) -> str:
        scheduler = WriteBatchScheduler.__new__(WriteBatchScheduler)
        scheduler.kind = "apply_code_fixes"
        scheduler.output_dir = str(self.out)
        scheduler.phase = "apply-code-fixes"
        scheduler.state = {}
        scheduler.tasks = self.tasks
        scheduler.plan_file = str(self.plan)
        task = self.tasks[index]
        return scheduler._with_result_contract(task, scheduler.dispatched_prompt(task))


SUMMARIES = [
    "Opponent progress unwraps only at the start-line crossing.",
    "Wrapped ranking progress within each lap; npm test passed.",
    "Gate targets now derive from the frame's starting progress.",
]


def test_the_earlier_batches_are_named_by_their_batch_ids(tmp_path):
    rig = _Racing08(tmp_path)
    for index, summary in enumerate(SUMMARIES):
        rig.native_append(index, summary)
    block = _block(rig.native_prompt(3))
    assert block.splitlines() == [
        f"- **apply-fix-batch-{index}** ({rig.tasks[index]['prior_changes_id']}): "
        f"{summary}"
        for index, summary in enumerate(SUMMARIES)
    ], block
    assert "Batch 1" not in block and "**Batch" not in block, block


def test_both_legs_bind_the_same_bytes(tmp_path):
    rig = _Racing08(tmp_path)
    for index, summary in enumerate(SUMMARIES):
        rig.native_append(index, summary)
    assert rig.native_prompt(3) == rig.executor_prompt(3)


def test_the_native_append_records_the_batch_id_from_the_task_file(tmp_path):
    rig = _Racing08(tmp_path)
    rig.native_append(1, SUMMARIES[1])
    (line,) = (rig.out / prior_changes.FILENAME).read_text().splitlines()
    entry = json.loads(line)
    assert entry["id"] == rig.tasks[1]["prior_changes_id"]
    assert entry["task_id"] == "apply-fix-batch-1"
    assert entry["phase"] == "apply-fixes"


def test_an_entry_no_task_file_names_is_listed_by_its_own_id(tmp_path):
    """An older log, or an id the phase's task file does not hold: the
    entry id alone, never a made-up batch number."""
    prior_changes.append(str(tmp_path), "index.html#a4d7fa43", "apply-fixes", "Did A.")
    assert prior_changes.render(str(tmp_path), phase="apply-fixes") == (
        "- **index.html#a4d7fa43**: Did A."
    )


def test_the_task_file_name_is_the_bridges():
    assert prior_changes.WRITE_TASK_FILENAME == WRITE_TASK_FILENAME


# -- the executor leg ------------------------------------------------------


def test_the_executor_names_the_batch_that_wrote_each_entry(env):  # noqa: F811
    """No task file on disk here: the scheduler passes its own task id."""
    prompts = {}

    def respond(task, prompt):
        prompts[task["task_id"]] = prompt
        n = task["task_id"].rsplit("-", 1)[1]
        env.plan.write_text(env.plan.read_text() + f"s{n}\n")
        return {"status": "applied", "ids": [f"s{n}"], "files_touched": ["plans/p.md"],
                "summary": f"applied s{n}"}

    assert not (env.phase_dir / WRITE_TASK_FILENAME).exists()
    result = env.scheduler(
        [env.apply_task(0), env.apply_task(1)], respond=respond,
        state=FakeState(env.tmp_path / "state.json"),
    ).run()
    assert result.exit_code == 0, result.messages
    entries = prior_changes.entries(str(env.out_dir))
    assert [(e["id"], e.get("task_id")) for e in entries] == [
        ("sec-0", "apply-batch-0"), ("sec-1", "apply-batch-1")]
    assert "- **apply-batch-0** (sec-0): applied s0" in prompts["apply-batch-1"]
    assert "**Batch" not in prompts["apply-batch-1"]


# -- an id printed once ----------------------------------------------------


def test_an_implement_entry_names_its_task_once(tmp_path):
    prior_changes.append(str(tmp_path), "T001", "implement", "Created index.html.")
    prior_changes.append(
        str(tmp_path), "T002", "implement",
        "T002: 8-point closed CatmullRom track, 400 samples with normals.")
    assert prior_changes.render(str(tmp_path), phase="implement").splitlines() == [
        "- **T001**: Created index.html.",
        "- **T002**: 8-point closed CatmullRom track, 400 samples with normals.",
    ]


def test_the_implementation_record_prints_a_task_id_once(tmp_path):
    """OpenCode racing 08: "- (T002): T002: 8-point closed CatmullRom track"."""
    prior_changes.append(
        str(tmp_path), "T002", "implement",
        "T002: 8-point closed CatmullRom track, 400 samples with normals.")
    prior_changes.append(
        str(tmp_path), "T005", "implement",
        "T005 — added pure findNearestSample window search in index.html")
    prompt = _fix_prompt(tmp_path)
    assert "- (T002): 8-point closed CatmullRom track, 400 samples with normals." in prompt
    assert "- (T005): added pure findNearestSample window search in index.html" in prompt
    assert "T002: 8-point" not in prompt and "T005 — added" not in prompt
    # The record line is still recognised as quoting the entry.
    assert sorted(prior_changes.quoted_in(prompt, str(tmp_path))) == [
        ("T002", "implement"), ("T005", "implement")]


def test_an_id_inside_the_summary_is_kept():
    strip = prior_changes.summary_without_ids
    assert strip("T002: did X", ["T002"]) == "did X"
    assert strip("(T002) did X", ["T002"]) == "did X"
    assert strip("[T002]: did X", ["T002"]) == "did X"
    assert strip("apply-fix-batch-0 - did X", ["apply-fix-batch-0"]) == "did X"
    # Not a leading label: kept word for word.
    assert strip("T002 and T003 share the track helpers", ["T002"]) == (
        "T002 and T003 share the track helpers")
    assert strip("T0021: another task", ["T002"]) == "T0021: another task"
    assert strip("Fixed T002: the lap rule", ["T002"]) == "Fixed T002: the lap rule"
    assert strip("T002:", ["T002"]) == "T002:"


def test_the_read_command_prints_the_render(tmp_path, capsys):
    prior_changes.append(str(tmp_path), "T001", "implement", "T001: Created a.js.")
    prior_changes.append(str(tmp_path), "sec#1", "apply-suggestions", "Plan edit.")
    assert prior_changes.main(["read", "--output-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.rstrip("\n") == prior_changes.render(str(tmp_path))
