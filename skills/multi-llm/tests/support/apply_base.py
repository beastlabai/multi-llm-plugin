"""``ApplyOrchestratorBase`` under test: a concrete mock subclass and its inputs.

``MockApplyOrchestrator`` implements every abstract method with a minimal stub
and records the calls; ``_make_args`` builds its namespace, and
``_setup_plan_with_output`` lays out a plan with a review-plan report.
``_FakeOutcome`` / ``_FakeSchedulerResult`` stand in for the write scheduler's
results, and ``_orchestrator_on(tmp_path)`` builds one over them. No fixtures.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.apply_orchestrator_base import ApplyOrchestratorBase


class MockApplyOrchestrator(ApplyOrchestratorBase):
    """Concrete mock subclass implementing all abstract methods with minimal stubs.

    Records method calls for verifying template method sequencing.
    """

    phase_name = "apply-mock"
    review_subdir = "review-plan"
    item_noun = "suggestion"
    supports_revalidation = False
    supports_skip_flag = False

    def __init__(self, args: argparse.Namespace):
        # Track method call order
        self.call_log: List[str] = []
        super().__init__(args)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        self.call_log.append("load_data")
        return ([], [])

    def parse_user_edits(self, report_path: str) -> Dict[str, Tuple[str, str]]:
        self.call_log.append("parse_user_edits")
        return {}

    def merge_user_edits(
        self,
        groups: List[Dict[str, Any]],
        edited_descriptions: Dict[str, Tuple[str, str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        self.call_log.append("merge_user_edits")
        return (groups, [])

    def parse_skips_from_report(
        self, report_path: str
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        self.call_log.append("parse_skips_from_report")
        return (set(), set(), set())

    def parse_validation_overrides_from_report(
        self, report_path: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        self.call_log.append("parse_validation_overrides_from_report")
        return ({}, {})

    def format_item_for_output(
        self, group: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        self.call_log.append("format_item_for_output")
        return {"index": index, **group}

    def create_batches(self, items: List[Dict[str, Any]]) -> List[Any]:
        self.call_log.append("create_batches")
        return []

    def generate_batch_prompts(self, batches: List[Any]) -> List[Any]:
        self.call_log.append("generate_batch_prompts")
        return batches

    def build_output_json(
        self,
        batches: List[Any],
        *,
        resume_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.call_log.append("build_output_json")
        return {"status": "ok", "batches": [], "phase": self.phase_name}

    def get_output_path(self) -> str:
        self.call_log.append("get_output_path")
        phase_dir = os.path.join(self.out_dir, self.phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        return os.path.join(phase_dir, "orchestrator_output.json")

    def print_text_summary(
        self, batches: List[Any], output_path: str
    ) -> None:
        self.call_log.append("print_text_summary")


def _make_args(**overrides) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for testing."""
    defaults = {
        "plan_file": "/tmp/test-plan.md",
        "yes": False,
        "force": False,
        "dry_run": False,
        "fresh": False,
        "resume": False,
        "approve_all": False,
        "include_low": False,
        "min_priority": None,
        "verbose": False,
        "approve_all_low": False,
        "skip_all_human": False,
        "approve_importance": None,
        "skip_human_review": False,
        "no_batch": False,
        "max_batch_size": 4,
        "batch_review_mode": "by-importance",
        "accept_stale_consolidation": False,
        "no_confirm": True,  # Skip confirmation prompts in tests
        "include_high": False,
        "agent_decide": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _setup_plan_with_output(tmp_path: Path, plan_name: str = "test-plan") -> Path:
    """Create a plan file and its output directory structure for testing."""
    plan_path = tmp_path / f"{plan_name}.md"
    plan_path.write_text("# Test Plan\n", encoding="utf-8")

    # Create the output directory structure
    out_dir = tmp_path / plan_name
    out_dir.mkdir(exist_ok=True)

    # Create review-plan subdirectory with report.md
    review_dir = out_dir / "review-plan"
    review_dir.mkdir(exist_ok=True)
    (review_dir / "report.md").write_text("# Review Report\n", encoding="utf-8")

    return plan_path


class _FakeOutcome:
    """The `TaskOutcome` fields the outcome summary reads, and no more."""

    def __init__(self, task_id, status="ok", write_outcome="applied",
                 duration_seconds=1.0, runner="mock:model", reason=None,
                 files_touched=None, output_path=None):
        self.task_id = task_id
        self.status = status
        self.write_outcome = write_outcome
        self.duration_seconds = duration_seconds
        self.runner = runner
        self.reason = reason
        # F7/F10: only read by `touched_files()` and the batch's own result
        # file (`notes`) — optional so every existing caller is unaffected.
        self.files_touched = files_touched or []
        self.output_path = output_path

    @property
    def ok(self):
        return self.status == "ok"


class _FakeSchedulerResult:
    def __init__(self, outcomes, report_path=None, stopped_at_batch=None,
                 verification=None):
        self.outcomes = outcomes
        self.report_path = report_path
        self.stopped_at_batch = stopped_at_batch
        self.verification = verification


def _orchestrator_on(tmp_path, **arg_overrides):
    """A MockApplyOrchestrator that has been through _setup on a real plan."""
    plan_path = _setup_plan_with_output(tmp_path)
    args = _make_args(plan_file=str(plan_path), **arg_overrides)
    orch = MockApplyOrchestrator(args)
    orch._setup()
    return orch
