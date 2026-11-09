"""review-plan and review-code behave alike where one review lane says they must.

The review base runs review-plan (and review-tasks); ``code_review_orchestrator``
runs review-code. Code health REVIEW-7 found the two had drifted: a fix made on
one lane did not reach the other, and nothing failed. Every scenario here runs on
both lanes, offline: a real git repository, real phase dirs, real grouping,
validation rounds and reports, with only the model slots stubbed
(``run_all_models`` on the base, ``run_all_reviews`` on review-code). A scenario
that passes on one lane and fails on the other is drift
(docs/development/review-results.md, "The shared review lane").

Each test names the row of REVIEW-7's drift table it holds.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence
from unittest.mock import patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import code_review_orchestrator as cro  # noqa: E402
import review_plan_orchestrator as rpo  # noqa: E402
from utils import detach  # noqa: E402
from utils import review_lane  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils import status_file  # noqa: E402
from utils.state_manager import CURRENT_FORMAT_VERSION, get_or_create_state  # noqa: E402
from tests.support.gitrepo import commit_file, git, init_repo  # noqa: E402

MODELS = ["cursor-agent:auto", "codex:gpt-5.5"]

#: Mutually dissimilar names, so each finding is its own group: six groups make
#: two validation batches.
NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
PLAN_TYPES = ["addition", "modification", "deletion", "clarification"]


# ---------------------------------------------------------------------------
# One lane, driven through its own entry point
# ---------------------------------------------------------------------------


class Lane:
    """One review lane on a fresh repository, driven through its ``main``."""

    #: The lane's mode word, its phase dir, the module whose ``main`` runs it
    #: and the module that crosses its detach boundary (``maybe_detach``).
    mode: str
    phase: str
    module: Any
    boundary: Any

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch):
        self.repo = init_repo(root / "repo", commits=[
            ("src/app.py", "def handler():\n    return 1\n", "base"),
        ])
        # Everything that reads git without a directory reads the process's.
        monkeypatch.chdir(self.repo)
        self.base_sha = git(self.repo, "rev-parse", "HEAD").strip()
        commit_file(self.repo, "src/app.py",
                    "def handler():\n    return compute(2)\n", "change")
        self.plan = self.repo / "feature.md"
        self.plan.write_text("# Plan\n\n## Step 1\n\nChange the handler.\n",
                             encoding="utf-8")
        state = get_or_create_state(self.plan)
        state.state["tracked_files"] = [
            {"path": "src/app.py", "action": "modified", "task_id": "1"}]
        state.state["head_at_start"] = self.base_sha
        state.save()

    # -- paths ---------------------------------------------------------------

    @property
    def phase_dir(self) -> Path:
        return Path(rob.get_phase_dir(self.plan, self.phase))

    @property
    def state_path(self) -> Path:
        return get_or_create_state(self.plan).state_file

    def status(self) -> Dict[str, Any]:
        return status_file.read_status(str(self.phase_dir))

    def json(self, name: str) -> Any:
        return json.loads((self.phase_dir / name).read_text(encoding="utf-8"))

    def groups(self, name: str) -> List[Dict[str, Any]]:
        data = self.json(name)
        return data["groups"] if isinstance(data, dict) else data

    def phase_completed(self) -> bool:
        return self.phase in (
            get_or_create_state(self.plan).state.get("phases_completed") or {})

    # -- runs ----------------------------------------------------------------

    def argv(self, *flags: str) -> List[str]:
        return ["--plan-file", str(self.plan), *flags]

    def run(self, *flags: str, patches: Sequence[Any] = ()) -> int:
        """The lane's ``main`` with *flags*; its exit code (a ``SystemExit`` too)."""
        with contextlib.ExitStack() as stack:
            for extra in patches:
                stack.enter_context(extra)
            try:
                code = asyncio.run(self.module.main(self.argv(*flags)))
            except SystemExit as stop:
                code = stop.code
        return int(code or 0)

    def fanout(self, outcomes: Dict[str, Any], *flags: str,
               bypassed: Sequence[str] = ()) -> int:
        """A fan-out whose slots end as *outcomes* (``spec -> findings`` or
        ``spec -> "reason"`` for a failed slot); *bypassed* ran unsandboxed."""
        models = list(outcomes)
        return self.run(
            "--models", *models, "--timeout", "120", "--max-parallel", "2", *flags,
            patches=(
                patch("utils.interactive.resolve_models", return_value=models),
                patch("utils.interactive.is_model_valid", return_value=True),
                self.stub_slots(outcomes, bypassed),
            ),
        )

    def reaggregate(self, *flags: str, patches: Sequence[Any] = ()) -> int:
        return self.run("--reaggregate", *flags, patches=patches)

    def write_findings(self, spec: str, findings: List[Dict[str, Any]]) -> None:
        """Replace *spec*'s result file, as a salvage or a hand edit would."""
        stem = rob.sanitize_model_name(spec)
        (self.phase_dir / f"{stem}.json").write_text(json.dumps(findings),
                                                     encoding="utf-8")

    # -- lane specifics ------------------------------------------------------

    def finding(self, index: int, name: str, importance: str) -> Dict[str, Any]:
        raise NotImplementedError

    def stub_slots(self, outcomes: Dict[str, Any], bypassed: Sequence[str]) -> Any:
        raise NotImplementedError

    def findings(self, names: Sequence[str], high: Sequence[str] = ()) -> List[Dict]:
        return [
            self.finding(index, name, "HIGH" if name in high else "MEDIUM")
            for index, name in enumerate(names)
        ]


class PlanLane(Lane):
    mode, phase, module, boundary = "review-plan", "review-plan", rpo, rob

    def finding(self, index, name, importance):
        return {
            "title": name,
            "desc": f"{name}{name} {name}{name}",
            "importance": importance,
            "reference": f"Step {index + 1}",
            "type": PLAN_TYPES[index % len(PLAN_TYPES)],
        }

    def stub_slots(self, outcomes, bypassed):
        async def run_all_models(*_args, **kwargs):
            tracker = kwargs.get("tracker")
            if tracker is not None:
                tracker.seed()
            results = {}
            for spec, outcome in outcomes.items():
                failed = isinstance(outcome, str)
                results[spec] = {
                    "success": not failed,
                    "output": None if failed else json.dumps(outcome),
                    "error": outcome if failed else None,
                    "prompt": "", "stdout": "", "stderr": "",
                    "duration_seconds": 0.0,
                    "model_spec": spec,
                    "source": "failed" if failed else "file",
                    **({"sandbox": "bypassed"} if spec in bypassed else {}),
                }
                if tracker is not None:
                    tracker.record_result(spec, results[spec])
            return results

        return patch.object(rob, "run_all_models", new=run_all_models)


class CodeLane(Lane):
    mode, phase, module, boundary = "review-code", "code-review", cro, cro

    def argv(self, *flags):
        return super().argv("--base-ref", self.base_sha, *flags)

    def finding(self, index, name, importance):
        return {
            "title": name,
            "desc": f"{name}{name} {name}{name}",
            "importance": importance.lower(),
            "file": "src/app.py",
            "line": 2,
            "type": f"kind-{name.lower()}",
        }

    def stub_slots(self, outcomes, bypassed):
        async def run_all_reviews(*_args, **kwargs):
            served = kwargs.get("served_models")
            for spec in bypassed:
                if served is not None:
                    served[spec] = {"sandbox": "bypassed"}
            return {
                spec: ((False, None, outcome) if isinstance(outcome, str)
                       else (True, outcome, None))
                for spec, outcome in outcomes.items()
            }

        return patch.object(cro, "run_all_reviews", new=run_all_reviews)


@pytest.fixture(params=[PlanLane, CodeLane], ids=["review-plan", "review-code"])
def lane(request, tmp_path, monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    return request.param(tmp_path, monkeypatch)


def _two_models(lane: Lane, *, high: Sequence[str] = ()) -> Dict[str, List[Dict]]:
    """Six findings split across two models, one shared."""
    return {
        MODELS[0]: lane.findings(NAMES[:4], high=high),
        MODELS[1]: lane.findings(NAMES[3:], high=high),
    }


def _round_handed_to_the_host(lane: Lane, **kwargs) -> Dict[str, Any]:
    """A fan-out left where the host takes over: grouped, round materialized."""
    assert lane.fanout(_two_models(lane, **kwargs)) == 0
    assert (lane.phase_dir / "validation_tasks.json").exists()
    return lane.json("validation_tasks.json")


def _write_batch(lane: Lane, batch: Dict[str, Any], status: str = "valid") -> None:
    verdicts = [
        {
            "group_index": global_index,
            "group_hash": batch["group_ids"][local_index],
            "status": status,
            "reason": f"Checked (group {global_index}).",
            "confidence": 0.9,
        }
        for local_index, global_index in enumerate(batch["group_indices"])
    ]
    (lane.phase_dir / f"validation_batch_{batch['batch_index']}.json").write_text(
        json.dumps({"groups": verdicts}), encoding="utf-8")


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------


class TestReaggregate:
    def test_records_the_high_groups(self, lane):
        """Row 1 (REVIEW-2): a reaggregate records the distinct HIGH groups
        where a detached caller reads them."""
        _round_handed_to_the_host(lane, high=["Alpha"])

        assert lane.reaggregate("--skip-validation") == 0

        assert lane.status()["high_priority_groups"] == {"count": 1}
        assert lane.phase_completed()

    def test_an_unfinished_round_is_neither_merged_nor_completed(self, lane, capsys):
        """Row 2 (REVIEW-3): with batch files missing, nothing is merged and
        the phase stays open, owing validation."""
        tasks = _round_handed_to_the_host(lane)
        assert tasks["total_batches"] >= 2
        _write_batch(lane, tasks["batches"][0])
        capsys.readouterr()

        assert lane.reaggregate() == 0

        out = capsys.readouterr().out
        assert f"Only 1/{tasks['total_batches']} batch files found." in out
        assert not (lane.phase_dir / "validation.json").exists()
        assert not lane.phase_completed()
        assert lane.status()["state"] == "validation_pending"

    def test_a_whole_round_is_merged_and_completed(self, lane):
        tasks = _round_handed_to_the_host(lane)
        for batch in tasks["batches"]:
            _write_batch(lane, batch)

        assert lane.reaggregate() == 0

        verdicts = lane.groups("validation.json")
        assert verdicts and all(v["status"] == "valid" for v in verdicts)
        assert lane.phase_completed()
        assert lane.status()["state"] == "complete"

    def test_skip_validation_reports_the_groups_on_disk(self, lane):
        """Row 3: ``--reaggregate --skip-validation`` puts the grouped findings
        in the report's data, not an empty list."""
        _round_handed_to_the_host(lane)

        assert lane.reaggregate("--skip-validation") == 0

        grouped = lane.groups("grouped.json")
        assert grouped
        assert len(lane.json("report_data.json")["groups"]) == len(grouped)

    def test_zero_groups_retire_the_previous_round(self, lane):
        """Row 6: results that hold no findings retire the round bound to the
        old grouping, and the empty verdicts take its place."""
        _round_handed_to_the_host(lane)
        for spec in MODELS:
            lane.write_findings(spec, [])

        assert lane.reaggregate() == 0

        assert lane.json("validation.json") == []
        assert not (lane.phase_dir / "validation_tasks.json").exists()
        assert list(lane.phase_dir.glob("validation_tasks.*.stale.json"))
        assert lane.phase_completed()

    def test_a_new_round_clears_the_previous_rounds_subagents(self, lane):
        """Rows 7 and 8: a round a reaggregate materializes clears the
        executor's count of the previous round and stamps its format version."""
        _round_handed_to_the_host(lane)
        (lane.phase_dir / "validation_tasks.json").unlink()
        detach.report_progress(str(lane.phase_dir),
                               subagents={"done": 16, "total": 16})

        assert lane.reaggregate() == 0

        tasks = lane.json("validation_tasks.json")
        assert tasks["format_version"] == CURRENT_FORMAT_VERSION
        progress = lane.status()["progress"]
        assert not progress.get("subagents")
        assert progress["step"] == "validation_pending"


class TestFanout:
    def test_the_round_records_its_format_version(self, lane):
        """Row 8: the fan-out's task file carries the format version."""
        tasks = _round_handed_to_the_host(lane)
        assert tasks["format_version"] == CURRENT_FORMAT_VERSION

    def test_the_host_hand_off_records_the_durable_follow_up(self, lane):
        """Row 9: the hand-off persists the follow-up as mode word, plan and
        flags, so ``status`` can name the command that advances the phase."""
        tasks = _round_handed_to_the_host(lane)

        status = lane.status()
        assert status["state"] == "validation_pending"
        assert status["reaggregate_mode"] == lane.mode
        assert Path(status["plan_file"]) == lane.plan
        assert isinstance(status["replay_flags"], list)
        assert Path(status["validation_tasks_path"]) == (
            lane.phase_dir / "validation_tasks.json")
        marker = ("VALIDATION_BATCHES_PENDING" if tasks["total_batches"] > 1
                  else "VALIDATION_PENDING")
        assert status["stdout_markers"] == [marker]
        command = detach.resolved_reaggregate_command(status)
        assert f" {lane.mode} " in command and "--reaggregate" in command

    def test_every_slot_failing_still_records_the_bypassed_sandbox(self, lane):
        """Row 5: a run whose every slot failed still records which slots ran
        without a sandbox, durably and for the poller."""
        code = lane.fanout(
            {MODELS[0]: "provider crashed", MODELS[1]: "provider crashed"},
            bypassed=[MODELS[0]],
        )

        assert code == 3
        status = lane.status()
        assert status["models_completed"] == []
        assert set(status["models_failed"]) == set(MODELS)
        assert status["models_served"][MODELS[0]]["sandbox"] == "bypassed"
        assert status["progress"]["sandbox_bypassed"] == [MODELS[0]]


class TestDetachLauncher:
    """Row 10: a ``--detach`` launcher writes no state before its boundary, so
    a launch refused there leaves ``state.json`` as it found it."""

    @staticmethod
    def _refused_launch(lane: Lane, *flags: str) -> int:
        return lane.run(
            "--models", *MODELS, "--detach", *flags,
            patches=(
                patch("utils.interactive.resolve_models", return_value=list(MODELS)),
                patch("utils.interactive.is_model_valid", return_value=True),
                patch.object(lane.boundary, "maybe_detach", return_value=3),
            ),
        )

    def test_a_first_run_writes_no_state_file(self, lane):
        os.unlink(lane.state_path)

        assert self._refused_launch(lane) == 3

        assert not lane.state_path.exists()

    def test_force_leaves_the_completion_to_the_worker(self, lane):
        state = get_or_create_state(lane.plan)
        state.mark_phase_completed(lane.phase)
        state.save()
        before = lane.state_path.read_bytes()

        assert self._refused_launch(lane, "--force") == 3

        assert lane.state_path.read_bytes() == before
        assert lane.phase_completed()


class TestSlotResult:
    """Both lanes' slot shapes read as one :class:`review_lane.SlotResult`."""

    def test_review_plans_record(self):
        record = {"success": False, "output": None, "error": "crashed",
                  "source": "failed", "stdout": "[{", "failure": "parse"}
        slot = review_lane.SlotResult.coerce(MODELS[0], record)
        assert (slot.success, slot.payload, slot.error) == (False, None, "crashed")
        assert slot.source == "failed" and slot.failure == "parse"
        assert slot.details is record

    def test_review_codes_triple(self):
        issues = [{"title": "Alpha"}]
        slot = review_lane.SlotResult.coerce(MODELS[1], (True, issues, None))
        assert (slot.success, slot.payload, slot.error) == (True, issues, None)
        assert slot.as_review() == (True, issues, None)

    def test_a_fanout_map_reads_whichever_shape_each_slot_has(self):
        slots = review_lane.slot_results({
            MODELS[0]: {"success": True, "output": "[]", "error": None},
            MODELS[1]: (False, None, "timed out"),
        })
        assert [s.model_spec for s in slots.values()] == MODELS
        assert slots[MODELS[0]].payload == "[]"
        assert slots[MODELS[1]].error == "timed out"
        assert review_lane.SlotResult.coerce(MODELS[0], slots[MODELS[0]]) is slots[MODELS[0]]
