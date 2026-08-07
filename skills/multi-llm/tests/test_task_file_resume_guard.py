"""The resume reader refuses a stale task file with exit 3 (D2g consumer sweep).

A phase dir is persisted on disk and re-read on resume, and with prompts embedded
a task file is a *snapshot*. A run started under one skill version and resumed
under the other would silently mis-spawn (an old reader ignores ``prompt`` and
re-derives it from markdown; a new reader finds no ``prompt`` and dispatches an
empty task), and a plan edited between materialization and resume leaves prompts
that still validate cleanly while reviewing text nobody is looking at any more.
Both fail loudly here rather than dispatching anything.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from utils import review_orchestrator_base as rob  # noqa: E402
from utils.state_manager import save_groups_payload, stamp_stable_ids  # noqa: E402
from utils.validation import prepare_batched_validation_tasks  # noqa: E402


def _phase(tmp_path):
    """A phase dir with one model result and a grouped.json, ready to reaggregate."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\nDo the thing.\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    phase_dir = out_dir / "review-plan"
    phase_dir.mkdir(parents=True)

    suggestions = [
        {
            "title": f"Finding {i}",
            "desc": "Some description",
            "importance": "MEDIUM",
            "reference": "Step 1",
            "type": "addition",
        }
        for i in range(3)
    ]
    (phase_dir / "model-a.json").write_text(json.dumps(suggestions), encoding="utf-8")

    grouped = [
        {
            "theme": s["title"],
            "category": "x",
            "models": ["model-a"],
            "suggestions": [s],
        }
        for s in suggestions
    ]
    stamp_stable_ids(grouped)
    (phase_dir / "grouped.json").write_text(
        json.dumps(save_groups_payload(grouped)), encoding="utf-8"
    )
    return plan, out_dir, phase_dir, grouped


def _args(**over):
    base = dict(
        skip_validation=False,
        validation_model="auto",
        timeout=None,
        internal_validation=False,
        force=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _reaggregate(plan, out_dir, phase_dir):
    return asyncio.run(
        rob.reaggregate_from_existing_files(
            prefix="plan",
            out_dir=str(out_dir),
            phase_dir=str(phase_dir),
            phase_name="review-plan",
            plan_path=str(plan),
            args=_args(),
            orchestrator_script="review_plan_orchestrator.py",
        )
    )


class TestResumeReaderRefusals:
    def test_v1_task_file_exits_3_naming_the_phase_dir(self, tmp_path, capsys):
        plan, out_dir, phase_dir, _grouped = _phase(tmp_path)
        # Today's implicit reference-based shape: no version, no prompts.
        (phase_dir / "validation_tasks.json").write_text(
            json.dumps({
                "batches": [{"batch_index": 0, "group_indices": [0, 1, 2],
                             "output_path": str(phase_dir / "validation_batch_0.json")}],
                "total_batches": 1,
                "grouped_file": str(phase_dir / "grouped.json"),
                "plan_file": str(plan),
            }),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as excinfo:
            _reaggregate(plan, out_dir, phase_dir)
        assert excinfo.value.code == 3

        err = capsys.readouterr().err
        assert "version 1" in err
        assert str(phase_dir) in err
        assert "--force" in err

    def test_matching_v2_task_file_is_accepted(self, tmp_path, capsys):
        plan, out_dir, phase_dir, grouped = _phase(tmp_path)
        document = prepare_batched_validation_tasks(
            groups=grouped,
            context=plan.read_text(encoding="utf-8"),
            output_dir=str(phase_dir),
            plan_file=str(plan),
        )
        (phase_dir / "validation_tasks.json").write_text(
            json.dumps(document), encoding="utf-8"
        )

        # No batch results yet, so the reader proceeds and reports the shortfall
        # instead of refusing the file.
        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out
        assert "batch files found" in out

    def test_edited_plan_exits_3_naming_the_changed_input(self, tmp_path, capsys):
        plan, out_dir, phase_dir, grouped = _phase(tmp_path)
        document = prepare_batched_validation_tasks(
            groups=grouped,
            context=plan.read_text(encoding="utf-8"),
            output_dir=str(phase_dir),
            plan_file=str(plan),
        )
        (phase_dir / "validation_tasks.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        # The plan moves between materialization and resume.
        plan.write_text("# Plan\n\nActually do something else entirely.\n",
                        encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            _reaggregate(plan, out_dir, phase_dir)
        assert excinfo.value.code == 3

        err = capsys.readouterr().err
        assert "plan_sha256" in err
        assert "Nothing was dispatched" in err

    def test_regrouped_phase_exits_3(self, tmp_path, capsys):
        plan, out_dir, phase_dir, grouped = _phase(tmp_path)
        document = prepare_batched_validation_tasks(
            groups=grouped,
            context=plan.read_text(encoding="utf-8"),
            output_dir=str(phase_dir),
            plan_file=str(plan),
        )
        (phase_dir / "validation_tasks.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        (phase_dir / "grouped.json").write_text(
            json.dumps(save_groups_payload(grouped[:1])), encoding="utf-8"
        )

        with pytest.raises(SystemExit) as excinfo:
            _reaggregate(plan, out_dir, phase_dir)
        assert excinfo.value.code == 3
        assert "grouped_sha256" in capsys.readouterr().err

    def test_future_version_exits_3(self, tmp_path, capsys):
        plan, out_dir, phase_dir, _grouped = _phase(tmp_path)
        (phase_dir / "validation_tasks.json").write_text(
            json.dumps({"tasks_schema_version": 99, "batches": []}), encoding="utf-8"
        )
        with pytest.raises(SystemExit) as excinfo:
            _reaggregate(plan, out_dir, phase_dir)
        assert excinfo.value.code == 3
        assert "99" in capsys.readouterr().err


class TestHarnessSimulatesTheInvokerHalf:
    """``SkillRunner.simulate_subagents`` plays the D3a print/persist contract."""

    def test_writes_one_result_per_task_and_refuses_a_v1_file(self, tmp_path):
        from tests.harness.skill_runner import SkillRunner

        plan, _out_dir, phase_dir, grouped = _phase(tmp_path)
        document = prepare_batched_validation_tasks(
            groups=grouped,
            context=plan.read_text(encoding="utf-8"),
            output_dir=str(phase_dir),
            plan_file=str(plan),
            max_per_batch=2,
        )
        task_file = phase_dir / "validation_tasks.json"
        task_file.write_text(json.dumps(document), encoding="utf-8")

        written = SkillRunner.simulate_subagents(task_file)
        assert len(written) == document["total_batches"]
        for path in written:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["metadata"]["schema_version"] == "2.1"
            assert all(g["group_hash"] for g in payload["groups"])

        v1 = phase_dir / "v1_tasks.json"
        v1.write_text(json.dumps({"batches": []}), encoding="utf-8")
        with pytest.raises(AssertionError, match="tasks_schema_version 2"):
            SkillRunner.simulate_subagents(v1)
