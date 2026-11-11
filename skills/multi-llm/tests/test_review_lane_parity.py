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
import hashlib
import io
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
from utils import cli_common  # noqa: E402
from utils import detach  # noqa: E402
from utils import review_lane  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils import review_round  # noqa: E402
from utils import status_file  # noqa: E402
from utils.progress_tracker import FanoutProgress  # noqa: E402
from utils.state_manager import (  # noqa: E402
    CURRENT_FORMAT_VERSION, StateManager, get_or_create_state,
)
from utils.subagent_tasks import build_inputs  # noqa: E402
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
    #: Whether reused verdicts are held to the task file's inputs
    #: (``ReviewLane.reuse_checks_inputs``), and the input an edit to one
    #: reviewed line moves.
    reuse_checks_inputs: bool
    reviewed_input: str
    #: Whether a plain re-run over files a round left on disk is judged by a
    #: partial-completion guard (review-code's: refused, or its completion
    #: finished) rather than run again (the base's).
    guards_partial_rounds: bool

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

    def fanout_slots(self, outcomes: Dict[str, Any], *flags: str) -> int:
        """A fan-out through the lane's real fan-out, with only each slot
        function stubbed: its findings, or the exception it raises."""
        models = list(outcomes)
        return self.run(
            "--models", *models, "--timeout", "120", "--max-parallel", "2", *flags,
            patches=(
                patch("utils.interactive.resolve_models", return_value=models),
                patch("utils.interactive.is_model_valid", return_value=True),
                self.stub_slot(outcomes),
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

    def edit_a_reviewed_line(self) -> None:
        """Change one line of what the lane's findings were reviewed against."""
        raise NotImplementedError

    def stub_slots(self, outcomes: Dict[str, Any], bypassed: Sequence[str]) -> Any:
        raise NotImplementedError

    def stub_slot(self, outcomes: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def findings(self, names: Sequence[str], high: Sequence[str] = ()) -> List[Dict]:
        return [
            self.finding(index, name, "HIGH" if name in high else "MEDIUM")
            for index, name in enumerate(names)
        ]

    # -- one slot (``TestTheSlotCore``) ---------------------------------------

    #: Where the lanes' slots legitimately differ (``ReviewSlot``): review-code's
    #: slot takes its terminal tracker row itself, puts a failure's spend on its
    #: error and writes a salvage request itself; the base's slot catches an
    #: exception raised past its deadline and records it.
    slot_takes_terminal_rows: bool
    slot_spend_on_error: bool
    slot_catches_its_own_errors: bool
    slot_writes_salvage_requests: bool
    #: Whether the report's hand-back names review-code's ``issues.json``.
    hands_back_issues: bool
    #: What a completed fan-out and a completed reaggregate write and print
    #: from the report on, in order (``TestTheCompletionTail``).
    fanout_tail: Sequence[str]
    reaggregate_tail: Sequence[str]
    #: The flags that make a HIGH group exit the lane (review-code's
    #: ``--fail-on-high``; the base lane has no gate) and the exit they give.
    gate_flags: Sequence[str]
    gate_exit: int

    def tracker(self) -> FanoutProgress:
        """A seeded one-model fan-out map in the phase dir."""
        self.phase_dir.mkdir(parents=True, exist_ok=True)
        tracker = FanoutProgress(str(self.phase_dir), [MODELS[0]],
                                 anchor=str(self.plan), timeout=30)
        tracker.seed()
        return tracker

    def slot(self, invoke: Any, *, tracker: Any = None) -> review_lane.SlotResult:
        """The lane's slot function on ``MODELS[0]``, each attempt answered by
        *invoke* (its ``invoke_with_file_output``), as one ``SlotResult``."""
        self.phase_dir.mkdir(parents=True, exist_ok=True)
        with patch.object(self.slot_module(), "invoke_with_file_output", invoke), \
             patch.object(rob.asyncio, "sleep", _no_sleep):
            outcome = asyncio.run(self.slot_call(tracker))
        return review_lane.SlotResult.coerce(MODELS[0], outcome)

    def slot_module(self) -> Any:
        raise NotImplementedError

    def slot_class(self) -> type:
        raise NotImplementedError

    def slot_call(self, tracker: Any) -> Any:
        raise NotImplementedError

    def answer(self, outcome: review_lane.SlotResult) -> Any:
        """The findings a completed slot returned."""
        raise NotImplementedError

    def save_slot(self, outcome: review_lane.SlotResult) -> bool:
        """Save the slot as the fan-out's save loop does."""
        raise NotImplementedError


class PlanLane(Lane):
    mode, phase, module, boundary = "review-plan", "review-plan", rpo, rob
    reuse_checks_inputs, reviewed_input = False, "plan_sha256"
    guards_partial_rounds = False
    slot_takes_terminal_rows = slot_spend_on_error = False
    slot_catches_its_own_errors, slot_writes_salvage_requests = True, False
    hands_back_issues = False
    gate_flags, gate_exit = (), 0
    fanout_tail = (
        "report.md", "report.html", "artifacts", "high_priority_groups",
        "state.json: phase recorded", "state: complete", "step: complete",
        "teardown", "=== Sandbox and jail checks", "exit 0",
    )
    reaggregate_tail = (
        "report.md", "report.html", "artifacts", "high_priority_groups",
        "validation_counts", "Reaggregation complete!", "Report:",
        "[consolidation]", "state.json: phase recorded", "state: complete",
        "step: complete", "teardown", "exit 0",
    )

    def slot_module(self):
        return rob

    def slot_class(self):
        return rob.BaseSlot

    def slot_call(self, tracker):
        return rob.run_single_model(
            MODELS[0], str(self.plan), SLOT_TEMPLATE, {}, "review-plan",
            "plan_review", 30, out_dir=str(self.phase_dir), prefix="feature",
            tracker=tracker, run_model_specs=[MODELS[0]],
        )

    def answer(self, outcome):
        return json.loads(outcome.payload)

    def save_slot(self, outcome):
        return rob.save_model_result(
            "feature", outcome.model_spec, outcome.success, outcome.payload,
            outcome.error, str(self.phase_dir), str(self.phase_dir), "review_plan",
            failure_output=rob.failure_output_from_result(outcome.details),
            failure=outcome.failure,
        )

    def edit_a_reviewed_line(self):
        self.plan.write_text("# Plan\n\n## Step 1\n\nChange the handler to 999.\n",
                             encoding="utf-8")

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

    def stub_slot(self, outcomes):
        async def run_single_model(model_spec, *_args, **_kwargs):
            outcome = outcomes[model_spec]
            if isinstance(outcome, BaseException):
                raise outcome
            return {
                "success": True, "output": json.dumps(outcome), "error": None,
                "prompt": "", "stdout": "", "stderr": "",
                "duration_seconds": 0.0, "model_spec": model_spec,
                "source": "file",
            }

        return patch.object(rob, "run_single_model", new=run_single_model)


class CodeLane(Lane):
    mode, phase, module, boundary = "review-code", "code-review", cro, cro
    reuse_checks_inputs, reviewed_input = True, "reviewed_code_sha256"
    guards_partial_rounds = True
    slot_takes_terminal_rows = slot_spend_on_error = True
    slot_catches_its_own_errors, slot_writes_salvage_requests = False, True
    hands_back_issues = True
    gate_flags, gate_exit = ("--fail-on-high",), 3
    fanout_tail = (
        "report.html", "report.md", "Report saved to:", "artifacts",
        "state.json: phase recorded", "state: complete", "step: complete",
        "teardown", "=== Sandbox and jail checks", "intent-to-add exit",
        "high_priority_groups", "=== Review Summary ===",
        "[HIGH_PRIORITY_ISSUES]", "exit 3",
    )
    reaggregate_tail = (
        "validation_counts", "report.html", "report.md", "Reaggregation complete!",
        "Report:", "artifacts", "state.json: phase recorded", "state: complete",
        "step: complete", "high_priority_groups", "=== Review Summary ===",
        "[HIGH_PRIORITY_ISSUES]", "intent-to-add exit", "teardown", "exit 3",
    )

    def slot_module(self):
        return cro

    def slot_class(self):
        return cro.CodeSlot

    def slot_call(self, tracker):
        return cro.run_single_review(
            MODELS[0], self.plan, self.phase_dir, timeout=30, tracker=tracker,
            usage={}, run_model_specs=[MODELS[0]],
        )

    def answer(self, outcome):
        return outcome.payload

    def save_slot(self, outcome):
        return cro.save_model_result(
            "feature", outcome.model_spec, outcome.success, outcome.payload,
            outcome.error, self.phase_dir,
        )

    def edit_a_reviewed_line(self):
        (self.repo / "src/app.py").write_text("def handler():\n    return 999\n",
                                             encoding="utf-8")

    def reviewed_code(self) -> str:
        """The reviewed-code excerpt the round's prompts embed, cut from the
        grouping on disk and the files as they are now."""
        return cro.build_reviewed_code_excerpt(
            self.groups("grouped.json"), project_root=str(self.repo))

    def argv(self, *flags):
        return super().argv("--base-ref", self.base_sha, *flags)

    def finding(self, index, name, importance):
        return {
            "title": name,
            "desc": f"{name}{name} {name}{name}",
            "importance": importance.lower(),
            "file": "src/app.py",
            # A range, so the validator's excerpt holds the cited line
            # (`build_reviewed_code_excerpt` cuts its windows from ranges).
            "line_range": [2, 2],
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

    def stub_slot(self, outcomes):
        async def run_single_review(model_spec, *_args, **_kwargs):
            outcome = outcomes[model_spec]
            if isinstance(outcome, BaseException):
                raise outcome
            return (True, outcome, None)

        return patch.object(cro, "run_single_review", new=run_single_review)


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(params=[PlanLane, CodeLane], ids=["review-plan", "review-code"])
def lane(request, tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    return request.param(tmp_path, monkeypatch)


@pytest.fixture
def code_lane(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    return CodeLane(tmp_path, monkeypatch)


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


def _validated(lane: Lane, *flags: str, high: Sequence[str] = ()) -> Dict[str, Any]:
    """A round validated whole and merged by a first reaggregate; its task file."""
    assert lane.fanout(_two_models(lane, high=high), *flags) == 0
    tasks = lane.json("validation_tasks.json")
    for batch in tasks["batches"]:
        _write_batch(lane, batch)
    assert lane.reaggregate() == 0
    return tasks


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


#: A ``grouped.json`` that is not the grouping its verdicts bind: deleted, cut
#: short, or replaced with an empty grouping.
UNCHECKED_GROUPINGS = ["missing", "malformed", "empty"]


def _unchecked_grouping(lane: Lane, grouping: str) -> None:
    path = lane.phase_dir / "grouped.json"
    if grouping == "missing":
        path.unlink()
    elif grouping == "malformed":
        path.write_text("{", encoding="utf-8")
    else:
        path.write_text("[]", encoding="utf-8")


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

    def test_an_unfinished_round_hands_back_the_command_its_hand_off_named(
        self, lane, capsys,
    ):
        """The partial stop tells a host reading only the result line what to
        do: the missing batches are named, and ``next`` is the
        ``--reaggregate`` the round's first hand-off put there, not ``null``
        (docs/development/review-results.md, "Snapshots and empty review
        outcomes")."""
        # The dispatcher's result context, where `next` is recorded.
        first = cli_common.ResultContext(lane.mode, lane.phase)
        partial = cli_common.ResultContext(lane.mode, lane.phase)
        with patch.object(cli_common, "_ACTIVE", first):
            tasks = _round_handed_to_the_host(lane)
        _write_batch(lane, tasks["batches"][0])
        capsys.readouterr()

        with patch.object(cli_common, "_ACTIVE", partial):
            assert lane.reaggregate() == 0

        out = capsys.readouterr().out
        assert first.next and "--reaggregate" in first.next
        assert partial.next == first.next
        assert f"validation_batch_{tasks['batches'][1]['batch_index']}.json" in out
        assert "validation_batch_0.json," not in out.split("Missing:", 1)[1].split("\n", 1)[0]
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

    def test_a_bound_validation_json_is_reused_not_merged_again(self, lane, capsys):
        """A second reaggregate with nothing changed reuses the
        ``validation.json`` bound to the grouping beside it: its bytes and the
        grouping's stay, and the batches are not merged again. On review-code
        the round's reviewed-code excerpt is a real one, and still hashes to
        what the task file recorded."""
        tasks = _validated(lane)
        merged = (lane.phase_dir / "validation.json").read_bytes()
        grouped = (lane.phase_dir / "grouped.json").read_bytes()
        capsys.readouterr()

        assert lane.reaggregate() == 0

        out = capsys.readouterr().out
        assert "Using validation.json (bound to grouping " in out
        assert "batch files found, merging" not in out
        assert (lane.phase_dir / "validation.json").read_bytes() == merged
        assert (lane.phase_dir / "grouped.json").read_bytes() == grouped
        assert build_inputs(grouped_file=str(lane.phase_dir / "grouped.json"))[
            "grouped_sha256"] == tasks["inputs"]["grouped_sha256"]
        if isinstance(lane, CodeLane):
            excerpt = lane.reviewed_code()
            assert "return compute(2)" in excerpt
            assert build_inputs(reviewed_code=excerpt)["reviewed_code_sha256"] == (
                tasks["inputs"]["reviewed_code_sha256"])
        assert lane.phase_completed()
        assert lane.status()["state"] == "complete"

    def test_bound_verdicts_without_their_task_file(self, lane):
        """A dir that kept ``validation.json`` and its batch files but lost the
        task file. The base lane keeps its verdicts, which its round's
        snapshots bind and none of which are left to verify. review-code's
        are bound to the reviewed code, which nothing on disk records any
        more: they are not reused, and a new round is built, as with no
        ``validation.json`` at all."""
        _validated(lane)
        merged = (lane.phase_dir / "validation.json").read_bytes()
        (lane.phase_dir / "validation_tasks.json").unlink()

        assert lane.reaggregate() == 0

        if lane.reuse_checks_inputs:
            assert lane.json("validation_tasks.json")["format_version"] == (
                CURRENT_FORMAT_VERSION)
            assert lane.status()["state"] == "validation_pending"
            assert not lane.phase_completed()
        else:
            assert not (lane.phase_dir / "validation_tasks.json").exists()
            assert (lane.phase_dir / "validation.json").read_bytes() == merged
            assert lane.phase_completed()

    def test_a_rewritten_batch_retires_the_merged_verdicts(self, lane, capsys):
        """A batch file rewritten since the merge retires ``validation.json``
        (renamed, never deleted) and the round is merged again."""
        tasks = _round_handed_to_the_host(lane)
        for batch in tasks["batches"]:
            _write_batch(lane, batch)
        assert lane.reaggregate() == 0
        _write_batch(lane, tasks["batches"][0], status="invalid")
        capsys.readouterr()

        assert lane.reaggregate() == 0

        out = capsys.readouterr().out
        assert "Not using validation.json: " in out
        assert list(lane.phase_dir.glob("validation.*.stale.json"))
        statuses = {v["status"] for v in lane.groups("validation.json")}
        assert statuses == {"valid", "invalid"}

    def test_a_result_that_landed_after_grouping_is_regrouped(self, lane, capsys):
        """A finding that reached a result file after grouping (a salvage, an
        out-of-band result) is not dropped by reusing the bound grouping: the
        round is retired and a new one built over the regrouped findings."""
        tasks = _round_handed_to_the_host(lane)
        grouped_before = len(lane.groups("grouped.json"))
        lane.write_findings(MODELS[0], lane.findings([*NAMES[:4], "Golfgolf"]))
        capsys.readouterr()

        assert lane.reaggregate() == 0

        out = capsys.readouterr().out
        assert "grouped.json does not cover the model results on disk" in out
        assert list(lane.phase_dir.glob("validation_tasks.*.stale.json"))
        assert len(lane.groups("grouped.json")) == grouped_before + 1
        assert lane.json("validation_tasks.json")["batches"] != tasks["batches"]
        assert not lane.phase_completed()

    @pytest.mark.parametrize("grouping", ["missing", "empty"])
    def test_a_lost_grouping_is_regrouped_never_counted_as_zero(self, lane, grouping):
        """A reaggregate over a validated round whose completion was lost and
        whose ``grouped.json`` was then deleted or emptied regroups the results
        on disk: the phase is either completed with its HIGH group recorded or
        left open, never completed with none."""
        _validated(lane, high=["Alpha"])
        TestAPlainRerunOverARoundOnDisk._lose_the_completion(lane)
        _unchecked_grouping(lane, grouping)

        assert lane.reaggregate() == 0

        assert lane.groups("grouped.json")
        if lane.phase_completed():
            assert lane.status()["high_priority_groups"] == {"count": 1}

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


    def test_unreadable_results_clear_the_verdict_counts(self, lane, capsys):
        """A reaggregate that can read no result file ends before any report
        is rebuilt, so the verdict counts an earlier round recorded are
        cleared there (§6 T6.1), and the ending is named alike."""
        _validated(lane)
        assert lane.status().get("validation_counts")
        for spec in MODELS:
            stem = rob.sanitize_model_name(spec)
            (lane.phase_dir / f"{stem}.json").write_text("not json", encoding="utf-8")
        capsys.readouterr()

        assert lane.reaggregate() == 0

        out = capsys.readouterr().out
        assert "No readable model results to aggregate." in out
        assert "validation_counts" not in lane.status()

    def test_no_result_files_are_named_alike(self, lane, capsys):
        """A reaggregate over a phase dir with no result file at all names
        the dir it looked in, on both lanes."""
        _round_handed_to_the_host(lane)
        for spec in MODELS:
            (lane.phase_dir / f"{rob.sanitize_model_name(spec)}.json").unlink()
        capsys.readouterr()

        assert lane.reaggregate() == 0

        out = capsys.readouterr().out
        assert f"No model result files found in: {lane.phase_dir}" in out


class TestReusedVerdicts:
    """A reaggregate that reuses bound verdicts keeps each lane's freshness
    check (``ReviewLane.reuse_checks_inputs``): review-code's verdicts stay
    bound to the code they were reached on, the base lane's to its round's
    snapshots."""

    def test_a_reviewed_line_edited_after_validation(self, lane, capsys):
        """review-code refuses (exit 3) with the refusal a resume gives, naming
        the moved input; the base lane's verdicts are bound to the round's
        plan snapshot, which an edit to the live plan leaves as it was, so
        they are reused. Neither rewrites the verdicts."""
        tasks = _validated(lane)
        merged = (lane.phase_dir / "validation.json").read_bytes()
        lane.edit_a_reviewed_line()
        capsys.readouterr()

        code = lane.reaggregate()

        captured = capsys.readouterr()
        assert (lane.phase_dir / "validation.json").read_bytes() == merged
        if not lane.reuse_checks_inputs:
            assert code == 0
            assert "Using validation.json (bound to grouping " in captured.out
            return
        assert code == 3
        assert "Refusing task file " in captured.err
        assert "the inputs it was materialized from have changed" in captured.err
        recorded = tasks["inputs"][lane.reviewed_input]
        now = build_inputs(reviewed_code=lane.reviewed_code())[lane.reviewed_input]
        assert recorded != now
        assert (
            f"{lane.reviewed_input} (a reviewed file changed at the lines these "
            f"issues cite): materialized from {recorded!r}, now {now!r}"
        ) in captured.err
        assert "Using validation.json" not in captured.out

    def test_the_plan_edited_after_validation(self, lane, capsys):
        """review-code's task file binds the plan as it was materialized
        (``plan_sha256``), and a reused round is held to it as a resume is;
        the base lane reuses the verdicts its snapshot binds."""
        _validated(lane)
        lane.plan.write_text("# Plan\n\n## Step 1\n\nEdited since.\n",
                             encoding="utf-8")
        capsys.readouterr()

        code = lane.reaggregate()

        captured = capsys.readouterr()
        if lane.reuse_checks_inputs:
            assert code == 3
            assert "plan_sha256 (" in captured.err
        else:
            assert code == 0
            assert "Using validation.json (bound to grouping " in captured.out

    def test_a_commit_reviews_frozen_context_plan_edited_is_refused(
            self, code_lane, capsys):
        """A commit review's frozen context plan is in every prompt: a
        snapshot that no longer hashes to what the round recorded refuses the
        reaggregate (exit 3), reused verdicts or not."""
        snapshot = code_lane.repo / "context-plan.md"
        snapshot.write_text("# Context\nThe handler must return compute(2).\n",
                            encoding="utf-8")
        head = git(code_lane.repo, "rev-parse", "HEAD").strip()
        state = get_or_create_state(code_lane.plan)
        state.state["commit_range"] = {
            "base": code_lane.base_sha, "head": head, "brief": str(code_lane.plan),
            "commits": [{"sha": head, "subject": "change", "date": "2026-11-09",
                         "merge": False, "outside_window": False}],
            "plan_snapshot": str(snapshot),
            "plan_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "plan": "design.md", "drifted": [], "notes": [],
        }
        state.save()
        tasks = _validated(code_lane)
        assert tasks["inputs"]["context_plan_sha256"] == (
            hashlib.sha256(snapshot.read_bytes()).hexdigest())
        snapshot.write_text("# Changed context\nThe handler must return 999.\n",
                            encoding="utf-8")
        capsys.readouterr()

        assert code_lane.reaggregate() == 3

        captured = capsys.readouterr()
        assert "Refusing task file " in captured.err
        assert "context_plan_sha256 (" in captured.err

    def test_a_pinned_reviews_verdicts_ignore_the_working_tree(self, code_lane):
        """A review pinned to a commit (``--head-ref``) cuts its excerpts at
        that head, so a working-tree edit moves no input and the verdicts are
        kept."""
        head = git(code_lane.repo, "rev-parse", "HEAD").strip()
        tasks = _validated(code_lane, "--head-ref", head)
        code_lane.edit_a_reviewed_line()

        assert code_lane.reaggregate() == 0

        verdicts = code_lane.groups("validation.json")
        assert len(verdicts) == sum(len(b["group_ids"]) for b in tasks["batches"])
        assert all(v["status"] == "valid" for v in verdicts)

    def test_a_task_file_that_records_no_reviewed_code_hash(self, code_lane, capsys):
        """A task file that records no reviewed-code hash binds no verdict to
        the code: they are not reused, and its batch files are merged again,
        as with no ``validation.json``."""
        _validated(code_lane)
        task_path = code_lane.phase_dir / "validation_tasks.json"
        document = json.loads(task_path.read_text(encoding="utf-8"))
        del document["inputs"]["reviewed_code_sha256"]
        task_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        capsys.readouterr()

        assert code_lane.reaggregate() == 0

        out = capsys.readouterr().out
        assert "validation_tasks.json records no reviewed_code_sha256" in out
        assert "Using validation.json" not in out
        assert "batch files found, merging" in out
        assert all(v["status"] == "valid" for v in code_lane.groups("validation.json"))


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

    def test_the_grouping_is_recorded(self, lane):
        """The fan-out records its grouping's size where ``status`` reads it
        (``grouping_complete``): every model's findings, and the groups."""
        _round_handed_to_the_host(lane)

        status = lane.status()
        assert status["total_groups"] == len(lane.groups("grouped.json"))
        assert status["total_suggestions"] == len(NAMES) + 1

    def test_the_hand_off_describes_the_round_alike(self, lane, capsys):
        """The lines a host reads before it runs the batches are the same on
        both lanes, from the task file's own batching stats."""
        tasks = _round_handed_to_the_host(lane)

        out = capsys.readouterr().out
        stats = tasks["batching_stats"]
        assert f"Batches: {tasks['total_batches']}" in out
        assert stats["efficiency_gain_percent"] > 0
        assert f"Efficiency gain: {stats['efficiency_gain_percent']}%" in out
        assert f"Groups to validate: {stats['total_groups']}" in out
        assert f"HIGH priority groups: {stats['high_count']}" in out
        assert f"Normal priority groups: {stats['normal_count']}" in out

    def test_zero_findings_say_so(self, lane, capsys):
        """A fan-out whose models found nothing writes the empty grouping and
        says it did."""
        assert lane.fanout({spec: [] for spec in MODELS}) == 0

        out = capsys.readouterr().out
        assert "saved to: " in out and " (no " in out
        assert lane.groups("grouped.json") == []
        assert lane.json("validation.json") == []
        assert lane.phase_completed()

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


class TestAPlainRerunOverARoundOnDisk:
    """A plain re-run (no ``--force``) over the files an earlier round left
    never records the phase complete on a round that did not finish, and a
    phase it does record complete went through the completion every other
    ending runs (the HIGH groups recorded, ``--fail-on-high`` applied).

    The lanes reach that differently, on purpose: review-code's
    partial-completion guard refuses (exit 3, naming ``--reaggregate``) or,
    when the round's report is whole and sealed and only its record is
    missing, finishes that completion; the base runs the review again
    (docs/development/review-results.md, "The shared review lane")."""

    def test_a_round_handed_to_the_host_stays_open(self, lane):
        _round_handed_to_the_host(lane)

        code = lane.fanout(_two_models(lane))

        assert code == (3 if lane.guards_partial_rounds else 0)
        assert not lane.phase_completed()
        assert lane.status()["state"] != "complete"

    def test_a_round_with_some_verdicts_stays_open(self, lane):
        tasks = _round_handed_to_the_host(lane)
        _write_batch(lane, tasks["batches"][0])

        code = lane.fanout(_two_models(lane))

        assert code == (3 if lane.guards_partial_rounds else 0)
        assert not lane.phase_completed()
        assert lane.status()["state"] != "complete"

    @staticmethod
    def _lose_the_completion(lane: Lane) -> None:
        """The shape a run leaves when it stops after publishing its report
        and before recording the phase: no completion, no HIGH record."""
        state = get_or_create_state(lane.plan)
        state.state["phases_completed"].pop(lane.phase)
        state.save()
        status_file.remove_status_keys(str(lane.phase_dir), ["high_priority_groups"])

    def test_a_report_whose_completion_was_lost(self, lane, capsys):
        _validated(lane, high=["Alpha"])
        self._lose_the_completion(lane)
        capsys.readouterr()

        assert lane.fanout(_two_models(lane, high=["Alpha"])) == 0

        out = capsys.readouterr().out
        if lane.guards_partial_rounds:
            assert "Auto-healed" in out
            assert lane.phase_completed()
        else:
            assert not lane.phase_completed()
            assert lane.status()["state"] == "validation_pending"
        if lane.phase_completed():
            assert lane.status()["high_priority_groups"] == {"count": 1}
            assert "[HIGH_PRIORITY_ISSUES] 1" in out

    def test_a_healed_review_keeps_its_gate(self, code_lane):
        """``--fail-on-high`` holds on a review whose completion the guard
        finished: exit 3, ``modes.REVIEW_GATE``, with the phase recorded."""
        _validated(code_lane, high=["Alpha"])
        self._lose_the_completion(code_lane)

        assert code_lane.fanout(_two_models(code_lane, high=["Alpha"]),
                                "--fail-on-high") == 3

        assert code_lane.phase_completed()
        assert code_lane.status()["state"] == "complete"
        assert code_lane.status()["high_priority_groups"] == {"count": 1}

    @pytest.mark.parametrize("grouping", UNCHECKED_GROUPINGS)
    def test_a_lost_completion_over_a_grouping_that_does_not_check_out(
        self, lane, grouping, capsys,
    ):
        """A sealed report and its verdicts beside a ``grouped.json`` that is
        not the grouping those verdicts bind never complete the phase: the
        grouping is never read as zero groups. review-code's guard refuses
        (exit 3, naming the file and ``--force``) and records nothing; the
        base runs the review again."""
        _validated(lane, high=["Alpha"])
        self._lose_the_completion(lane)
        _unchecked_grouping(lane, grouping)
        before = lane.status()
        capsys.readouterr()

        code = lane.fanout(_two_models(lane, high=["Alpha"]))

        out = capsys.readouterr().out
        assert not lane.phase_completed()
        assert "Auto-healed" not in out
        assert "high_priority_groups" not in lane.status()
        if lane.guards_partial_rounds:
            assert code == 3
            assert "grouped.json" in out and "--force" in out
            assert lane.status() == before
        else:
            assert code == 0
            assert lane.status()["state"] == "validation_pending"

    @pytest.mark.parametrize("grouping", UNCHECKED_GROUPINGS)
    def test_a_healed_gate_never_counts_an_unchecked_grouping_as_zero(
        self, code_lane, grouping,
    ):
        """``--fail-on-high`` over a lost completion whose ``grouped.json``
        was deleted, cut or replaced with ``[]``: exit 3 from the refusal, the
        phase not recorded, no HIGH count recorded (the intact grouping is
        ``test_a_healed_review_keeps_its_gate``)."""
        _validated(code_lane, high=["Alpha"])
        self._lose_the_completion(code_lane)
        _unchecked_grouping(code_lane, grouping)

        assert code_lane.fanout(_two_models(code_lane, high=["Alpha"]),
                                "--fail-on-high") == 3

        assert not code_lane.phase_completed()
        assert "high_priority_groups" not in code_lane.status()


class TestFanoutSlots:
    """The fan-out's own mechanics, run for real on both lanes; only the slot
    functions (``run_single_model``, ``run_single_review``) are stubbed."""

    def test_a_slot_that_raises_is_failed_with_its_own_error(self, lane):
        """Row 11: an exception that ends a slot is that slot's failure, named
        with its own text and never as a deadline loss; the other model's
        findings are kept."""
        code = lane.fanout_slots({
            MODELS[0]: RuntimeError("boom"),
            MODELS[1]: lane.findings(NAMES[:2]),
        })

        assert code == 0
        status = lane.status()
        assert status["models_completed"] == [MODELS[1]]
        assert status["models_failed"] == {MODELS[0]: "Exception: RuntimeError: boom"}
        entry = status["progress"]["models"][MODELS[0]]
        assert (entry["state"], entry["source"]) == ("failed", "exception")
        assert (lane.phase_dir / "grouped.json").exists()

    def test_a_slot_cancelled_on_its_own_leaves_the_run_going(self, lane):
        """A slot whose own coroutine was cancelled is left to the sweep and
        the deadline backfill; the fan-out still returns the other slots."""
        code = lane.fanout_slots({
            MODELS[0]: asyncio.CancelledError(),
            MODELS[1]: lane.findings(NAMES[:2]),
        })

        assert code == 0
        status = lane.status()
        assert status["models_completed"] == [MODELS[1]]
        assert MODELS[0] in status["models_failed"]
        assert status["progress"]["models"][MODELS[0]]["state"] == "failed"

    def test_every_slot_reports_into_one_map(self, lane):
        """Each slot's terminal row comes from its own ending, done with the
        result's source; the counters agree with the map."""
        code = lane.fanout_slots({
            MODELS[0]: lane.findings(NAMES[:2]),
            MODELS[1]: lane.findings(NAMES[2:4]),
        })

        assert code == 0
        progress = lane.status()["progress"]
        assert {spec: row["state"] for spec, row in progress["models"].items()} == {
            MODELS[0]: "done", MODELS[1]: "done"}
        assert (progress["models_done"], progress["models_succeeded"]) == (2, 2)


#: The base slot's template: every placeholder its dispatch and record fill.
SLOT_TEMPLATE = "{prefix}{output_json_path}{output_location}{working_limits}"
#: What a failed attempt spent, reported by its adapter.
SPENT = {"output_tokens": 1234, "cost_usd": 0.5}
#: A failed attempt's leftover stdout that still holds an answer.
LEFT_BEHIND = '[{"title": "Alpha", "desc": "AlphaAlpha", "importance": "MEDIUM"}'


def _answered(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """An attempt that completed with *findings*."""
    return {"success": True, "data": findings, "source": "file",
            "output_file": "", "details": {"stderr": ""}}


def _lost(**details: Any) -> Dict[str, Any]:
    """An attempt that failed with *details*."""
    return {"success": False, "error": "provider exited 1",
            "details": {"stderr": "provider exited 1", **details}}


async def _no_sleep(_delay):
    return None


class TestTheReportsHandBack:
    """A published report's files are handed back on both channels by one
    helper (``review_lane.hand_back_report``) on every completion site;
    review-code's ``issues.json`` is the one file only it writes."""

    @staticmethod
    def _handed(lane: Lane, run) -> List[List[str]]:
        handed: List[List[str]] = []
        with patch.object(rob, "record_artifacts",
                          lambda paths: handed.append([Path(p).name for p in paths])):
            run()
        return handed

    def _expected(self, lane: Lane, *, validated: bool) -> List[str]:
        return (["report.md", "report.html", "grouped.json"]
                + (["validation.json"] if validated else [])
                + (["issues.json"] if lane.hands_back_issues else []))

    def test_a_fanouts_report(self, lane):
        handed = self._handed(lane, lambda: lane.fanout(_two_models(lane), "--skip-validation"))
        assert handed == [self._expected(lane, validated=False)]

    def test_a_reaggregates_report(self, lane):
        handed = self._handed(lane, lambda: _validated(lane))
        assert handed == [self._expected(lane, validated=True)]


#: The checkpoint keys a completion writes, traced when their value changes.
TAIL_KEYS = ("validation_counts", "high_priority_groups")
#: The lines a completion prints, traced by a phrase each holds.
TAIL_LINES = (
    "Report saved to:", "Reaggregation complete!", "Report:", "[consolidation]",
    "Auto-healed", "=== Review Summary ===", "[HIGH_PRIORITY_ISSUES]",
    "=== Sandbox and jail checks",
)
#: The closing-summary line each run is seeded with, so the summary has
#: something to print (it prints nothing for a run that kept no lines).
CLOSING_LINE = "[JAIL_DENIED] codex: simulated"
#: Where a completion's trace starts: the first thing a renderer writes.
TAIL_START = ("validation_counts", "report.html", "report.md", "artifacts")


class _TailLines:
    """A stdout that passes everything through and traces :data:`TAIL_LINES`
    and the phase ticker's teardown line, whole lines only."""

    def __init__(self, stream: Any, events: List[str]):
        self.stream, self.events, self.pending = stream, events, ""

    def write(self, text: str) -> int:
        self.pending += text
        *lines, self.pending = self.pending.split("\n")
        for line in lines:
            if line.startswith("[PROGRESS]") and "step: complete" in line:
                self.events.append("teardown")
                continue
            for phrase in TAIL_LINES:
                if phrase in line:
                    self.events.append(phrase)
                    break
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


class TestTheCompletionTail:
    """What a successful round writes and prints from its report on, in order,
    on each lane's completion sites, up to the process's exit code.

    The pieces are shared (``hand_back_report``, ``record_phase_complete``,
    ``complete_reaggregate``); the order around them is each lane's own, by
    decision (code health REVIEW-7): the base's per-group renderer publishes
    ``report.md`` before ``report.html``, hands the files back and records the
    HIGH groups and verdict counts before the phase is recorded complete;
    review-code's per-issue renderer records the verdict counts and writes
    ``report.html`` before publishing ``report.md``, and its HIGH-group record
    and ``--fail-on-high`` gate (``finish_code_review``, exit 3 here) follow the
    completion: on the fan-out after the ticker's teardown line, the closing
    summary and the ``intent_to_add_untracked`` exit, on a reaggregate before
    that exit and the teardown line. Every lane records the phase in
    ``state.json`` before the checkpoint's ``state: complete``.

    Traced: the renderer's writes, the hand-back, the phase record (the
    ``state.json`` save after ``mark_phase_completed``), the checkpoint's
    ``state``, ``step`` and HIGH keys, the teardown line, a seeded closing
    summary, the ``intent_to_add_untracked`` exit and the exit code. A change
    to any of these orders fails here and is then deliberate
    (docs/development/review-results.md, "Why each lane keeps its completion
    tail"). Not covered: a round that fails, is handed to the host, or runs
    with validation dispatched in-process."""

    @staticmethod
    def _trace(lane: Lane, run) -> List[str]:
        """The events of one completion, from the first thing its renderer
        writes to the process's exit code (``exit N``, what *run* returns)."""
        events: List[str] = []
        last = dict(lane.status())
        write_in_plan_dir = status_file.write_in_plan_dir
        record_artifacts = rob.record_artifacts
        mark_phase_completed = StateManager.mark_phase_completed
        save = StateManager.save
        intent_to_add_untracked = cro.intent_to_add_untracked
        marked: set = set()

        def traced_write(path, data, **kwargs):
            nonlocal last
            if Path(path).name == status_file.STATUS_FILENAME:
                document = json.loads(data)
                events.extend(key for key in TAIL_KEYS
                              if document.get(key) != last.get(key))
                if document.get("state") != last.get("state"):
                    events.append(f"state: {document.get('state')}")
                step = (document.get("progress") or {}).get("step")
                if step != (last.get("progress") or {}).get("step"):
                    events.append(f"step: {step}")
                last = document
            return write_in_plan_dir(path, data, **kwargs)

        def traced(event: str, real: Any) -> Any:
            def call(*args, **kwargs):
                events.append(event)
                return real(*args, **kwargs)
            return call

        def traced_mark(self, phase, *args, **kwargs):
            result = mark_phase_completed(self, phase, *args, **kwargs)
            if phase == lane.phase:
                marked.add(id(self))
            return result

        def traced_save(self, *args, **kwargs):
            result = save(self, *args, **kwargs)
            if id(self) in marked:
                marked.discard(id(self))
                events.append("state.json: phase recorded")
            return result

        @contextlib.contextmanager
        def traced_intent_to_add(*args, **kwargs):
            with intent_to_add_untracked(*args, **kwargs) as touched:
                yield touched
            events.append("intent-to-add exit")

        # What the run's slots kept for its closing summary.
        rob.note_for_closing_summary(lane.phase_dir, CLOSING_LINE)
        with patch.object(status_file, "write_in_plan_dir", traced_write), \
             patch.object(review_round, "publish_report",
                          traced("report.md", review_round.publish_report)), \
             patch.object(rob, "write_html_report",
                          traced("report.html", rob.write_html_report)), \
             patch.object(cro, "write_html_report",
                          traced("report.html", cro.write_html_report)), \
             patch.object(rob, "record_artifacts",
                          traced("artifacts", record_artifacts)), \
             patch.object(StateManager, "mark_phase_completed", traced_mark), \
             patch.object(StateManager, "save", traced_save), \
             patch.object(cro, "intent_to_add_untracked", traced_intent_to_add), \
             contextlib.redirect_stdout(_TailLines(sys.stdout, events)):
            code = run()
        # A run that prints no closing summary (a reaggregate, the heal) leaves
        # the seeded line behind; it is drained, unprinted, for the next test.
        with contextlib.redirect_stdout(io.StringIO()):
            rob.print_closing_summary(lane.phase_dir)
        events.append(f"exit {code}")
        start = next(i for i, event in enumerate(events) if event in TAIL_START)
        return events[start:]

    def test_a_fanouts_completion(self, lane):
        trace = self._trace(lane, lambda: lane.fanout(
            _two_models(lane, high=["Alpha"]), "--skip-validation",
            *lane.gate_flags))

        assert trace == list(lane.fanout_tail)

    def test_a_reaggregates_completion(self, lane):
        tasks = _round_handed_to_the_host(lane, high=["Alpha"])
        for batch in tasks["batches"]:
            _write_batch(lane, batch)

        trace = self._trace(lane, lambda: lane.reaggregate(*lane.gate_flags))

        assert trace == list(lane.reaggregate_tail)

    def test_a_healed_completion_ends_as_a_reaggregate_does(self, code_lane):
        """review-code's heal of a lost completion hands the files back and
        records the phase before the gate, as its reaggregate does; it has no
        report to render and no ticker of its own."""
        _validated(code_lane, high=["Alpha"])
        TestAPlainRerunOverARoundOnDisk._lose_the_completion(code_lane)
        # The checkpoint as the reaggregate left it before its completion.
        status_file.update_status(str(code_lane.phase_dir), {
            "state": "validation_pending", "progress": {"step": "validation_pending"},
        })

        trace = self._trace(code_lane, lambda: code_lane.fanout(
            _two_models(code_lane, high=["Alpha"]), *code_lane.gate_flags))

        assert trace == [
            "artifacts", "state.json: phase recorded", "state: complete",
            "step: complete", "Auto-healed", "Report:", "high_priority_groups",
            "=== Review Summary ===", "[HIGH_PRIORITY_ISSUES]", "exit 3",
        ]


class TestTheSlotCore:
    """One model slot, run on both lanes through the one core
    (``review_lane.run_slot``: the guard, the ladder, the checks) with only
    the dispatch stubbed. The lanes' slots still end in their own shapes and
    keep the differences ``ReviewSlot`` lists; each case pins one."""

    def test_both_slot_functions_are_entry_points_over_the_core(self, lane):
        """``run_single_model`` and ``run_single_review`` hand their lane's
        slot to ``run_slot`` and translate the one ``SlotResult`` back."""
        seen = []
        findings = lane.findings(NAMES[:1])

        async def run_slot(slot, *, start_time):
            seen.append(type(slot))
            return review_lane.SlotResult(
                slot.model_spec, True, payload=findings, error=None, source="file",
                details={"success": True, "output": json.dumps(findings),
                         "error": None, "source": "file"},
            )

        with patch.object(review_lane, "run_slot", run_slot):
            outcome = lane.slot(None)

        assert seen == [lane.slot_class()]
        assert outcome.success and outcome.error is None
        assert lane.answer(outcome) == findings

    def test_a_completed_slot_returns_its_answer(self, lane):
        findings = lane.findings(NAMES[:2])
        outcome = lane.slot(lambda **_kw: _answered(findings))

        assert outcome.success and outcome.error is None
        assert lane.answer(outcome) == findings

    def test_a_refused_slot_dispatches_nothing(self, lane):
        """The guard's refusal ends the slot undispatched with the gate's own
        line. review-code's slot takes its terminal row itself; the base's is
        the fan-out's, from the record. Either way the map reads the same."""
        reason = "[SANDBOX_UNAVAILABLE] the runner's sandbox cannot start"
        calls = []
        tracker = lane.tracker()
        verdict = rob.SandboxVerdict(rob.SANDBOX_REFUSE, reason)
        with patch.object(rob, "sandbox_refusal", lambda *_a: verdict):
            outcome = lane.slot(lambda **kw: calls.append(kw), tracker=tracker)

        assert calls == []
        assert (outcome.success, outcome.error) == (False, reason)
        row = lane.status()["progress"]["models"][MODELS[0]]
        assert (row["state"] == "failed") is lane.slot_takes_terminal_rows
        # The fan-out's terminal transition from what the slot returned
        # (`run_models_concurrent`); idempotent on a row already terminal.
        tracker.record_result(MODELS[0], outcome)
        row = lane.status()["progress"]["models"][MODELS[0]]
        assert (row["state"], row["source"]) == ("failed", "not-dispatched")

    def test_a_failed_slot_names_its_spend_on_review_code_only(self, lane):
        """What a failed slot spent: review-code puts it on the error string
        every failure artifact carries (its triple has no other place); the
        base records it on the record's own ``usage`` key."""
        outcome = lane.slot(lambda **_kw: _lost(usage=dict(SPENT)))

        assert not outcome.success
        assert outcome.error.startswith("provider exited 1")
        assert ("(spent $0.50 est., 1,234 output tokens)" in outcome.error) is (
            lane.slot_spend_on_error)
        if not lane.slot_spend_on_error:
            assert outcome.details["usage"]["output_tokens"] == 1234

    def test_an_exception_past_the_deadline_is_the_slots_own_error(self, lane):
        """An exception between the slot's deadline and its ending: the base's
        slot records it itself (with the template as the prompt); review-code's
        raises to the fan-out's wrapper, which records it the same way
        (``raised_slot_record``). Both name the exception's own text."""
        boom = RuntimeError("ladder broke")

        def slot_ladder(*_args, **_kwargs):
            raise boom

        with patch.object(lane.slot_module(), "slot_ladder", slot_ladder):
            if lane.slot_catches_its_own_errors:
                outcome = lane.slot(lambda **_kw: _answered([]))
                assert outcome.details["prompt"] == SLOT_TEMPLATE
            else:
                with pytest.raises(RuntimeError, match="ladder broke"):
                    lane.slot(lambda **_kw: _answered([]))
                outcome = review_lane.SlotResult.from_record(
                    MODELS[0], rob.raised_slot_record(MODELS[0], boom, anchor=str(lane.plan)))

        assert not outcome.success
        assert outcome.error == "Exception: RuntimeError: ladder broke"

    def test_a_lost_answer_gets_one_salvage_request(self, lane):
        """A failed attempt whose leftover stdout holds an answer: review-code's
        slot writes the salvage request itself, the base's when the record is
        saved. Either way one request carries that text."""
        outcome = lane.slot(lambda **_kw: _lost(stdout_excerpt=LEFT_BEHIND))
        stem = rob.sanitize_model_name(MODELS[0])
        request = lane.phase_dir / f"salvage_{stem}.json"

        assert request.exists() is lane.slot_writes_salvage_requests
        lane.save_slot(outcome)
        assert json.loads(request.read_text(encoding="utf-8"))["raw_output"] == LEFT_BEHIND


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

    def test_the_phase_dirs_snapshot_is_left_to_the_worker(self, lane):
        """No copy of the reviewed document (review-plan's ``backup.md``) is
        written before the boundary either: a refused launch leaves the one a
        running round reads as it was."""
        lane.phase_dir.mkdir(parents=True, exist_ok=True)
        snapshot = lane.phase_dir / "backup.md"
        snapshot.write_bytes(b"# The plan the running round reviewed\n")
        lane.plan.write_text("# Plan\n\n## Step 1\n\nEdited since.\n",
                             encoding="utf-8")
        before = self._artifacts(lane)

        assert self._refused_launch(lane) == 3

        assert snapshot.read_bytes() == b"# The plan the running round reviewed\n"
        assert self._artifacts(lane) == before

    @staticmethod
    def _artifacts(lane: Lane) -> List[str]:
        """The phase dir's artifacts; the dot files (the run's lock) aside."""
        return sorted(p.name for p in lane.phase_dir.iterdir()
                      if not p.name.startswith("."))

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
