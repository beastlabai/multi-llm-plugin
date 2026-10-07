"""A consolidation that ran is never recommended again (live run 8, #7).

Claude Code racing 01: review-plan recommended consolidation ("17 valid
groups >= 15"), the host ran ``--consolidate`` and
``--reaggregate-consolidation`` in the same run, and afterwards ``status``
still printed "consolidation recommended: 17 valid groups >= 15 — run
--consolidate" and put the recommendation on its result line, through every
later phase. The checkpoint kept the key the pass had answered, and nothing
that read it asked whether the pass had already run.

"Ran" is the rule apply-suggestions already uses for consolidated decisions:
``consolidated.json`` built from the ``grouped.json`` beside it
(``metadata.grouped_hash``). Groups that changed since make it stale, and the
recommendation stands again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
for entry in (str(SKILL_ROOT), str(Path(__file__).parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import multi_llm  # noqa: E402
import review_plan_orchestrator as rpo  # noqa: E402
import full_workflow_orchestrator as fwo  # noqa: E402
import modes as mode_table  # noqa: E402
from mocks.mock_llm import echo_subagent_answer  # noqa: E402
from utils import cli_common, run_status, status_file  # noqa: E402
from utils.cli_common import record_next  # noqa: E402
from utils.consolidation import consolidation_is_current  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.review_orchestrator_base import (  # noqa: E402
    recompute_consolidation_recommendation,
)
from utils.state_manager import (  # noqa: E402
    StateManager, save_groups_payload, stamp_stable_ids,
)
from utils.subagent_executor import simulate_native_invoker  # noqa: E402

THRESHOLD = 2
LINE = "consolidation recommended:"


def _group(theme: str) -> dict:
    return {
        "theme": theme,
        "reference": "## Task 1: Authentication",
        "category": "modification",
        "importance": "MEDIUM",
        "models": ["model-a"],
        "suggestions": [{
            "title": f"{theme} suggestion", "desc": f"d for {theme}",
            "type": "modification", "importance": "MEDIUM",
        }],
    }


def _reviewed(tmp_path: Path, themes=("Auth validation", "Auth tokens", "Auth expiry")):
    """A finished review-plan whose checkpoint recommends consolidation."""
    plan = tmp_path / "plan.md"
    plan.write_text("# Plan\n\n## Task 1: Authentication\nDo auth.\n", encoding="utf-8")
    phase_dir = output_dir_path(plan) / "review-plan"
    phase_dir.mkdir(parents=True)
    groups = [_group(theme) for theme in themes]
    stamp_stable_ids(groups)
    (phase_dir / "grouped.json").write_text(
        json.dumps(save_groups_payload(groups), indent=2), encoding="utf-8"
    )
    (phase_dir / "validation.json").write_text(json.dumps({"groups": [
        {"group_index": index, "status": "valid", "reason": ""}
        for index in range(len(groups))
    ]}), encoding="utf-8")
    status_file.write_status(str(phase_dir), {
        "phase": "review-plan", "state": "complete",
        "consolidation_recommended": {
            "valid_groups": len(groups), "threshold": THRESHOLD,
            "command": "--consolidate",
        },
    })
    return plan, phase_dir, groups


def _consolidate_natively(plan: Path, phase_dir: Path) -> None:
    """The Claude Code native leg: the marker, the host's subagents, the merge."""
    rpo.consolidate_mode(str(phase_dir), str(plan))
    document = json.loads(
        (phase_dir / "consolidation_tasks.json").read_text(encoding="utf-8")
    )
    report = simulate_native_invoker(
        document["batches"], phase_dir=str(phase_dir), kind="consolidation",
        respond=lambda task: echo_subagent_answer("consolidation", task["prompt"]),
    )
    assert report.ok, report.failure_summary()
    rpo.reaggregate_consolidation_mode(str(phase_dir), str(plan), "plan", [])
    assert (phase_dir / "consolidated.json").exists()


def _status_run_view(plan: Path, capsys):
    capsys.readouterr()
    context = cli_common.activate_result(cli_common.ResultContext())
    try:
        run_status.main(["--plan-file", str(plan), "--view", "run"])
        recorded = context.consolidation_recommended
    finally:
        cli_common.deactivate_result()
    return capsys.readouterr().out, recorded


class TestTheConsolidationPassAnswersTheRecommendation:

    def test_the_merge_takes_the_recommendation_off_the_checkpoint(
        self, tmp_path, capsys
    ):
        plan, phase_dir, _ = _reviewed(tmp_path)
        _consolidate_natively(plan, phase_dir)
        status = status_file.read_status(str(phase_dir))
        assert "consolidation_recommended" not in status
        # Every other key of the checkpoint survives.
        assert status["state"] == "complete"

        out, recorded = _status_run_view(plan, capsys)
        assert LINE not in out
        assert recorded is None

    def test_status_drops_one_a_checkpoint_kept_from_before_the_pass(
        self, tmp_path, capsys
    ):
        """The evidence's own shape: a checkpoint written by a build without
        the rule above still holds the key after consolidation ran."""
        plan, phase_dir, _ = _reviewed(tmp_path)
        _consolidate_natively(plan, phase_dir)
        status_file.update_status(str(phase_dir), {
            "consolidation_recommended": {
                "valid_groups": 3, "threshold": THRESHOLD, "command": "--consolidate",
            },
        })
        out, recorded = _status_run_view(plan, capsys)
        assert LINE not in out
        assert recorded is None

    def test_a_reaggregate_after_it_does_not_recommend_it_again(
        self, tmp_path, capsys
    ):
        plan, phase_dir, groups = _reviewed(tmp_path)
        _consolidate_natively(plan, phase_dir)
        validated = [dict(g, validation_status="valid") for g in groups]
        args = argparse.Namespace(consolidation_threshold=THRESHOLD)
        context = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert recompute_consolidation_recommendation(
                str(phase_dir), validated, args, True
            ) is None
            assert context.consolidation_recommended is None
        finally:
            cli_common.deactivate_result()
        assert "[CONSOLIDATION_RECOMMENDED]" not in capsys.readouterr().out
        assert "consolidation_recommended" not in status_file.read_status(str(phase_dir))


class TestGroupsThatChangedSinceAreRecommendedAgain:

    def test_a_new_grouping_makes_the_consolidation_stale(self, tmp_path, capsys):
        plan, phase_dir, _ = _reviewed(tmp_path)
        _consolidate_natively(plan, phase_dir)
        assert consolidation_is_current(str(phase_dir))

        regrouped = [_group(t) for t in ("Auth validation", "Auth tokens", "Auth audit")]
        stamp_stable_ids(regrouped)
        (phase_dir / "grouped.json").write_text(
            json.dumps(save_groups_payload(regrouped), indent=2), encoding="utf-8"
        )
        assert not consolidation_is_current(str(phase_dir))

        validated = [dict(g, validation_status="valid") for g in regrouped]
        args = argparse.Namespace(consolidation_threshold=THRESHOLD)
        context = cli_common.activate_result(cli_common.ResultContext())
        try:
            recommendation = recompute_consolidation_recommendation(
                str(phase_dir), validated, args, True
            )
        finally:
            cli_common.deactivate_result()
        assert recommendation == {
            "valid_groups": 3, "threshold": THRESHOLD, "command": "--consolidate",
        }
        out, recorded = _status_run_view(plan, capsys)
        assert LINE in out
        assert recorded == recommendation

    @pytest.mark.parametrize("content", [
        None, "not json", json.dumps({"metadata": {}}),
        json.dumps({"metadata": {"grouped_hash": ""}}),
    ])
    def test_a_missing_or_unhashed_file_is_not_a_consolidation(self, tmp_path, content):
        _, phase_dir, _ = _reviewed(tmp_path)
        if content is not None:
            (phase_dir / "consolidated.json").write_text(content, encoding="utf-8")
        assert not consolidation_is_current(str(phase_dir))


class TestAResumedFullWorkflowDoesNotRunItTwice:
    """`--full`'s consolidation step is not skippable, so every resume of an
    unattended workflow re-entered it and, at the threshold, ran the pass on
    groups it had already consolidated."""

    def test_the_step_is_skipped_once_the_groups_are_consolidated(
        self, tmp_path, monkeypatch, capsys
    ):
        from utils.consolidation import CONSOLIDATION_RECOMMENDED_THRESHOLD

        calls = []

        def run_mode(self, mode_word, extra):
            calls.append((mode_word, list(extra)))
            phase = mode_table.phase_for(mode_word) or mode_word
            manager = StateManager(self.plan_path)
            manager.mark_phase_completed(phase)
            manager.save()
            record_next(None)
            return 0

        monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)
        plan = tmp_path / "wf-plan.md"
        plan.write_text("# Plan\n\n## Step 1\nDo it.\n", encoding="utf-8")
        state = StateManager(plan)
        state.mark_phase_completed("review-plan")
        state.save()
        phase_dir = output_dir_path(plan) / "review-plan"
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "validation.json").write_text(json.dumps({"groups": [
            {"group_index": i, "status": "valid"}
            for i in range(CONSOLIDATION_RECOMMENDED_THRESHOLD)
        ]}), encoding="utf-8")
        grouped = json.dumps({"groups": []})
        (phase_dir / "grouped.json").write_text(grouped, encoding="utf-8")
        (phase_dir / "consolidated.json").write_text(json.dumps({"metadata": {
            "grouped_hash": hashlib.sha256(grouped.encode("utf-8")).hexdigest(),
        }}), encoding="utf-8")

        multi_llm.main(["full", str(plan), "--yes", "--resume"])
        out = capsys.readouterr().out
        assert ("review-plan", ["--consolidate"]) not in [
            (mode, extra[:1]) for mode, extra in calls
        ], calls
        assert "already ran on these review groups" in out
