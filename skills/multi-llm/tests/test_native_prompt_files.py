"""Consolidation and task generation hand a native host prompt files (live run 8, #7).

Claude Code racing 01: `consolidation_tasks.json`'s batches carried no
`prompt_path`, so the host wrote each batch's prompt to
`/tmp/consol_prompt_N.txt` itself (outside the repository and the plan's
output folder) to hand its subagents a file. Racing 03: the one
`task_generation` task carried none either, so the host read the 21 KB prompt
into its own context and pasted it whole into the subagent call.

Both kinds now get the sidecar validation batches have had since 2026-09-19:
a file in the phase directory holding exactly the bytes of `prompt`, named on
the task as `prompt_path` with its line count, written all-or-none, with
`prompt` staying the source of truth; and the instructions tell the host to
hand the subagent that file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
for entry in (str(SKILL_ROOT), str(Path(__file__).parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import generate_tasks_orchestrator as gto  # noqa: E402
import review_plan_orchestrator as rpo  # noqa: E402
from mocks.mock_llm import echo_subagent_answer  # noqa: E402
from utils import validation as validation_mod  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import save_groups_payload, stamp_stable_ids  # noqa: E402
from utils.subagent_executor import simulate_native_invoker  # noqa: E402
from utils.subagent_tasks import (  # noqa: E402
    validate_subagent_task, validate_subagent_task_file,
)


#: The generate-tasks prompt file's name (``gto.PROMPT_FILE_NAME``), spelled
#: out so a tree without it fails each test rather than the collection.
PROMPT_FILE_NAME = "task_generation.prompt.txt"


def _flat(path: str) -> str:
    return " ".join((SKILL_ROOT / path).read_text(encoding="utf-8").split())


def _group(theme: str, section: str) -> dict:
    return {
        "theme": theme,
        "reference": section,
        "category": "modification",
        "importance": "MEDIUM",
        "models": ["model-a"],
        "suggestions": [{
            "title": f"{theme} suggestion", "desc": f"d for {theme}",
            "type": "modification", "importance": "MEDIUM",
        }],
    }


def _reviewed(tmp_path: Path, sections=("## Task 1: Auth", "## Task 2: Billing")):
    """A finished review whose groups make one consolidation batch per section."""
    plan = tmp_path / "plan.md"
    plan.write_text(
        "# Plan\n\n" + "".join(f"{s}\nDo it.\n\n" for s in sections), encoding="utf-8"
    )
    phase_dir = output_dir_path(plan) / "review-plan"
    phase_dir.mkdir(parents=True, exist_ok=True)
    groups = [
        _group(f"{section[3:]} {suffix}", section)
        for section in sections for suffix in ("validation", "tokens")
    ]
    stamp_stable_ids(groups)
    (phase_dir / "grouped.json").write_text(
        json.dumps(save_groups_payload(groups), indent=2), encoding="utf-8"
    )
    (phase_dir / "validation.json").write_text(json.dumps({"groups": [
        {"group_index": index, "status": "valid", "reason": ""}
        for index in range(len(groups))
    ]}), encoding="utf-8")
    return plan, phase_dir


def _consolidation_batches(phase_dir: Path):
    document = json.loads(
        (phase_dir / "consolidation_tasks.json").read_text(encoding="utf-8")
    )
    assert validate_subagent_task_file(document) == []
    return document["batches"]


class TestConsolidationBatchesCarryTheirPromptFile:

    def test_every_batch_names_a_file_in_the_phase_directory_holding_its_prompt(
        self, tmp_path, capsys
    ):
        plan, phase_dir = _reviewed(tmp_path)
        rpo.consolidate_mode(str(phase_dir), str(plan))
        batches = _consolidation_batches(phase_dir)
        assert len(batches) == 2
        for batch in batches:
            path = Path(batch["prompt_path"])
            assert path == phase_dir / (
                f"consolidation_batch_{batch['batch_index']}.prompt.txt"
            )
            with open(path, "r", encoding="utf-8", newline="") as handle:
                assert handle.read() == batch["prompt"]
            assert batch["prompt_lines"] == batch["prompt"].count("\n") + 1
            # The read instruction lives in host text, never in the prompt.
            assert "prompt.txt" not in batch["prompt"]
        out = capsys.readouterr().out
        assert "consolidation_batch_N.prompt.txt" in out

    def test_a_host_that_hands_its_subagents_the_files_completes_the_pass(
        self, tmp_path
    ):
        plan, phase_dir = _reviewed(tmp_path)
        rpo.consolidate_mode(str(phase_dir), str(plan))

        def from_the_file(task):
            text = Path(task["prompt_path"]).read_text(encoding="utf-8")
            return echo_subagent_answer("consolidation", text)

        report = simulate_native_invoker(
            _consolidation_batches(phase_dir), phase_dir=str(phase_dir),
            kind="consolidation", respond=from_the_file,
        )
        assert report.ok, report.failure_summary()
        rpo.reaggregate_consolidation_mode(str(phase_dir), str(plan), "plan", [])
        assert (phase_dir / "consolidated.json").exists()

    def test_a_narrower_round_leaves_no_prompt_file_of_the_wider_one(self, tmp_path):
        plan, phase_dir = _reviewed(tmp_path)
        rpo.consolidate_mode(str(phase_dir), str(plan))
        assert (phase_dir / "consolidation_batch_1.prompt.txt").exists()

        (tmp_path / "again").mkdir()
        _reviewed(tmp_path / "again", sections=("## Task 1: Auth",))
        narrower = output_dir_path(tmp_path / "again" / "plan.md") / "review-plan"
        for name in ("grouped.json", "validation.json"):
            (phase_dir / name).write_text(
                (narrower / name).read_text(encoding="utf-8"), encoding="utf-8"
            )
        rpo.consolidate_mode(str(phase_dir), str(plan))
        assert len(_consolidation_batches(phase_dir)) == 1
        assert not (phase_dir / "consolidation_batch_1.prompt.txt").exists()

    def test_a_file_that_cannot_be_written_leaves_every_batch_on_its_prompt(
        self, tmp_path, monkeypatch, capsys
    ):
        def cannot_write(target, prompt):
            raise OSError("disk full")

        monkeypatch.setattr(validation_mod, "_write_one_sidecar", cannot_write)
        plan, phase_dir = _reviewed(tmp_path)
        rpo.consolidate_mode(str(phase_dir), str(plan))
        for batch in _consolidation_batches(phase_dir):
            assert "prompt_path" not in batch and "prompt_lines" not in batch
            assert batch["prompt"]
        assert "[PROMPT_SIDECAR_SKIPPED]" in capsys.readouterr().err


class TestTaskGenerationCarriesItsPromptFile:

    def test_the_task_names_a_file_in_the_phase_directory_holding_its_prompt(
        self, tmp_path
    ):
        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n\n## Step 1\nBuild it.\n", encoding="utf-8")
        phase_dir = tmp_path / "plan" / "generate-tasks"
        task_file, document = gto.materialize_task_file(
            plan, str(phase_dir), repo_root=str(tmp_path)
        )
        task = json.loads(Path(task_file).read_text(encoding="utf-8"))["tasks"][0]
        assert validate_subagent_task(task) == []
        assert task["prompt_path"] == str(phase_dir / PROMPT_FILE_NAME)
        with open(task["prompt_path"], "r", encoding="utf-8", newline="") as handle:
            assert handle.read() == task["prompt"]
        assert task["prompt_lines"] == task["prompt"].count("\n") + 1
        assert "prompt.txt" not in task["prompt"]

    def test_the_hand_off_line_names_the_file(self, tmp_path, capsys):
        import multi_llm

        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n\n## Step 1\nBuild it.\n", encoding="utf-8")
        code = multi_llm.main(["generate-tasks", str(plan)])
        out = capsys.readouterr().out
        assert code == 0, out
        prompt_file = tmp_path / "plan" / "generate-tasks" / PROMPT_FILE_NAME
        assert prompt_file.exists()
        assert f"Run the prompt in {prompt_file}" in out


class TestTheHostIsToldToUseThem:

    def test_review_plan_hands_consolidation_subagents_the_file(self):
        text = _flat("instructions/review-plan.md")
        assert ("carries a complete `prompt`, and `prompt_path` "
                "(`consolidation_batch_N.prompt.txt` in the same directory") in text
        assert ("when the task carries `prompt_path`, the subagent reads that file "
                "in full (through line `prompt_lines`) and follows it") in text
        assert "write no prompt file of your own" in text

    def test_the_full_workflow_says_the_same(self):
        text = _flat("instructions/full-workflow.md")
        assert "its `prompt_path` file holds the same bytes" in text

    @pytest.mark.parametrize("phrase", [
        "have it read the file at `task.prompt_path`",
        "hand it the file at the task's `prompt_path` to read in full and follow "
        "(never paste the prompt into the subagent call)",
    ])
    def test_generate_tasks_hands_its_subagent_the_file(self, phrase):
        assert phrase in _flat("instructions/generate-tasks.md")
