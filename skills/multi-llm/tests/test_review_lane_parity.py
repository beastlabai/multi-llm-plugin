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


class PlanLane(Lane):
    mode, phase, module, boundary = "review-plan", "review-plan", rpo, rob
    reuse_checks_inputs, reviewed_input = False, "plan_sha256"
    guards_partial_rounds = False

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
