"""``--cascade`` on the executor leg: ``ImproveWorkflow``'s loop, one pass per stage.

Like ``tests/test_improve_orchestrator.py``, these tests stub
``FullWorkflow.run_mode``, so every argv a phase receives is asserted where the
composite hands it over. Unlike that file's stub, :class:`Phases` writes what a
real review and apply leave behind: the review dir's ``.status.json`` with the
stage's provenance block (read from the manifest its ``--cascade-stage-file``
names), ``grouped.json``, ``validation.json``, and the apply's
``write_tasks.json`` and ``state.json`` records. So the real
``improve_cascade.advance``, the archive's close-out, the forward ledger and
the per-round table all run on them; nothing of the cascade is faked.

The contract: docs/development/workflows.md, "A cascade (``--cascade``)".

Usage:
    uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_improve_cascade_executor.py -q
"""

import json
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import full_workflow_orchestrator as fwo  # noqa: E402
import modes as mode_table  # noqa: E402
import multi_llm  # noqa: E402
from code_review_orchestrator import REVIEW_SCOPE_FILES_KEY  # noqa: E402
from utils import cascade_archive  # noqa: E402
from utils import cascade_provenance as provenance  # noqa: E402
from utils import cli_common, detach, status_file  # noqa: E402
from utils import improve_cascade as ic  # noqa: E402
from utils import improve_workflow as iw  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils import unattended  # noqa: E402
from utils.cli_common import RESULT_MARKER, dispatcher_command  # noqa: E402
from utils.output_handler import PHASE_DIRECTORIES  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.subagent_tasks import content_hash  # noqa: E402

from tests.support.cascade_world import groups_of, member  # noqa: E402
from tests.support.improve_world import (  # noqa: E402
    APPLY_FLAGS,
    APPLY_PHASE,
    CODE,
    MODES,
    PLAN,
    REVIEW_PHASE,
    _main,
    _out,
    _plan,
    improve_repo,  # noqa: F401 - the `repo` fixture
)

A, B, C = "codex:gpt-a", "codex:gpt-b", "codex:gpt-c"
HIDDEN = provenance.HIDDEN_FLAG


class Killed(BaseException):
    """The process dying mid-step: nothing after the raise runs, no report, no stop."""


# ---------------------------------------------------------------------------
# The phases, fabricated
# ---------------------------------------------------------------------------

def _dir(plan_dir: Path, phase: str) -> Path:
    return plan_dir / PHASE_DIRECTORIES.get(phase, phase)


def _now() -> str:
    return datetime.now().isoformat()


class Phases:
    """``FullWorkflow.run_mode``, writing what each phase leaves.

    *zero* maps a model to the reason its review produced no result (row 3:
    ``models_completed: []``, exit 3 through ``report_partial_results``);
    *fail* maps ``(mode word, model)`` to an exit code returned with nothing
    written; *kill* is ``(mode word, model, when)``: ``"after"`` (the step
    finished, then the process died), ``"mid"`` (a review whose fan-out
    stamped its round and died), ``"started"`` (an apply that recorded its
    batches in progress and died). Every call is logged as ``(mode word,
    argv)``, and the composite run dir's ``cascade_stage`` as each step saw it.
    """

    def __init__(self, monkeypatch):
        self.calls: List[Tuple[str, List[str]]] = []
        self.stages_seen: List[Any] = []
        self.zero: Dict[str, str] = {}
        self.fail: Dict[Tuple[str, str], int] = {}
        self.kill: Optional[Tuple[str, str, str]] = None
        phases = self

        def run_mode(workflow, mode_word, extra):
            return phases.run(workflow, mode_word, list(extra))

        monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)

    # -- the dispatch ------------------------------------------------------
    def run(self, workflow, word: str, argv: List[str]) -> int:
        self.calls.append((word, argv))
        self.stages_seen.append(
            status_file.read_status(str(workflow.run_dir)).get("cascade_stage"))
        review, apply = workflow.row.sequence
        if word == review:
            model = argv[argv.index("--models") + 1]
            code = self.fail.get((word, model))
            if code is not None:
                return code
            code = self.review(workflow, argv, model)
        elif word == apply:
            model = self.reviewed_model(workflow)
            code = self.fail.get((word, model))
            if code is not None:
                return code
            if self.kill == (word, model, "started"):
                state = StateManager(workflow.plan_path)
                state.mark_phase_started(mode_table.phase_for(word))
                state.save()
                raise Killed()
            code = self.apply(workflow)
        else:  # pragma: no cover - a cascade runs no other step
            raise AssertionError(f"unexpected step {word}: {argv}")
        if self.kill == (word, model, "after"):
            raise Killed()
        return code

    @staticmethod
    def review_dir(workflow) -> Path:
        return _dir(workflow.plan_dir, workflow.row.sequence_phases[0])

    @staticmethod
    def apply_dir(workflow) -> Path:
        return _dir(workflow.plan_dir, workflow.row.sequence_phases[1])

    def reviewed_model(self, workflow) -> str:
        """The model of the round on record: its cascade block's (``""`` for a
        round outside a cascade)."""
        block = provenance.read_block(str(self.review_dir(workflow)))
        return block["model"] if block else ""

    # -- a review round ----------------------------------------------------
    def review(self, workflow, argv: List[str], model: str) -> int:
        review_phase = workflow.row.sequence_phases[0]
        directory = self.review_dir(workflow)
        code = workflow.row.sequence[0] == "review-code"
        if "--force" in argv:
            # A new round removes its own completion and retires the old
            # round's files (the step lock's `.lock` stays: the step holds it).
            state = StateManager(workflow.plan_path)
            state.state.get("phases_completed", {}).pop(review_phase, None)
            state.save()
            for path in directory.iterdir() if directory.is_dir() else ():
                if path.is_file() and not path.name.endswith(".lock"):
                    path.unlink()
        directory.mkdir(parents=True, exist_ok=True)
        status: Dict[str, Any] = {
            "phase": review_phase, "state": "models_running", "started_at": _now(),
            "round_id": secrets.token_hex(8), "models_requested": [model],
        }
        if HIDDEN in argv:
            manifest = Path(argv[argv.index(HIDDEN) + 1])
            status[provenance.STATUS_KEY] = provenance.block_from_manifest(manifest)
        if model in self.zero:
            status.update(models_completed=[], models_failed={model: self.zero[model]})
            status_file.write_status(str(directory), status)
            return unattended.report_partial_results(
                completed=[], failed=[model], phase_dir=str(directory))
        if self.kill == ("review-code" if code else "review-plan", model, "mid"):
            status_file.write_status(str(directory), status)
            raise Killed()
        finding = member(f"{model} finding", f"Something only {model} saw.", model=model,
                         file="src/app.py" if code else None)
        groups = groups_of([finding], code=code, model=model)
        grouped_text = json.dumps({"format_version": 2, "groups": groups}, indent=2)
        (directory / "grouped.json").write_text(grouped_text, encoding="utf-8")
        (directory / "validation.json").write_text(json.dumps({
            "format_version": 2,
            "groups": [{"group_index": i, "group_id": g["group_hash"], "status": "valid",
                        "reason": "real", "confidence": 0.9} for i, g in enumerate(groups)],
            "metadata": {"model": "host", "grouped_sha256": content_hash(grouped_text)},
        }, indent=2), encoding="utf-8")
        (directory / "report.md").write_text(f"# Review by {model}\n", encoding="utf-8")
        status.update(
            state="complete", models_completed=[model], models_failed={},
            validation_counts={"valid": len(groups), "invalid": 0,
                               "needs_human_decision": 0, "validation_failed": 0},
            usage={model: {"cost_usd": 0.01, "output_tokens": 100}},
        )
        status_file.write_status(str(directory), status)
        state = StateManager(workflow.plan_path)
        if code:
            listing = argv[argv.index("--files-from") + 1] if "--files-from" in argv else None
            if listing:
                state.state[REVIEW_SCOPE_FILES_KEY] = Path(listing).read_text(
                    encoding="utf-8").split()
            else:
                state.state.pop(REVIEW_SCOPE_FILES_KEY, None)
        state.mark_phase_completed(review_phase)
        state.save()
        return 0

    # -- its apply -----------------------------------------------------------
    def apply(self, workflow) -> int:
        apply_phase = workflow.row.sequence_phases[1]
        grouped_text = (self.review_dir(workflow) / "grouped.json").read_text(encoding="utf-8")
        sha = content_hash(grouped_text)
        ids = [group["group_hash"] for group in json.loads(grouped_text)["groups"]]
        directory = self.apply_dir(workflow)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "write_tasks.json").write_text(json.dumps({
            "schema_version": 1,
            "batches": [{"task_id": "b1", "expected_ids": ids, "declared_files": ["src/app.py"],
                         "inputs": {"grouped_sha256": sha}, "kind": "agent-write"}],
        }, indent=2), encoding="utf-8")
        (directory / "summary.md").write_text("# Applied\n", encoding="utf-8")
        state = StateManager(workflow.plan_path)
        processed = state.state.setdefault(f"processed_{directory.name}", {})
        for item in ids:
            processed[item] = {"status": "applied", "timestamp": _now(),
                               "details": {"task_id": "b1", "grouped_sha256": sha,
                                           "leg": "executor"}}
        if workflow.row.sequence[0] != "review-code":
            with open(workflow.plan_path, "a", encoding="utf-8") as handle:
                handle.write(f"\nApplied {ids[0][:8]}.\n")
        state.mark_phase_completed(apply_phase)
        state.save()
        return 0

    # -- reading the log -----------------------------------------------------
    def words(self) -> List[str]:
        return [word for word, _argv in self.calls]

    def reviews(self) -> List[List[str]]:
        return [argv for word, argv in self.calls if word.startswith("review-")]

    def applies(self) -> List[List[str]]:
        return [argv for word, argv in self.calls if word.startswith("apply-")]


@pytest.fixture
def phases(monkeypatch):
    return Phases(monkeypatch)


def _refuse_the_boundary(monkeypatch) -> None:
    """Fail the test if the composite reaches its detach boundary from here on."""
    def refuse(**kwargs):
        pytest.fail(f"maybe_detach was reached: {kwargs}")

    monkeypatch.setattr(rob, "maybe_detach", refuse)


@pytest.fixture
def dispatcher_env(monkeypatch):
    for name in ("MULTI_LLM_HOST", mode_table.ROUTE_ENV, detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(plan: Path, mode: str) -> Dict[str, Any]:
    current = ic.load_current(plan, mode)
    assert current is not None
    return current.record


def _cascade_dir(plan: Path, mode: str) -> Path:
    current = ic.load_current(plan, mode)
    assert current is not None
    return current.cascade_dir


def _manifest(plan: Path, mode: str, stage: int) -> str:
    return str(_cascade_dir(plan, mode) / f"stage-{stage}" / "inputs" / provenance.MANIFEST_NAME)


def _result_lines(out: str) -> List[Dict[str, Any]]:
    return [json.loads(line[len(RESULT_MARKER):]) for line in out.splitlines()
            if line.startswith(RESULT_MARKER)]


def _one_result(out: str) -> Dict[str, Any]:
    lines = _result_lines(out)
    assert len(lines) == 1, lines
    return lines[0]


def _continuation(mode: str, plan: Path, *flags: str) -> str:
    return dispatcher_command(mode, plan, *flags)


def _stage_states(plan: Path, mode: str) -> List[str]:
    return [stage["state"] for stage in _record(plan, mode)["stages"]]


def _tree(root: Path) -> Dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(root).parts}


# ---------------------------------------------------------------------------
# 1. Each stage's argv
# ---------------------------------------------------------------------------

class TestEachStagesArgv:
    @pytest.mark.parametrize("mode", MODES)
    def test_three_stages_each_review_one_model_then_apply(self, repo, phases, capsys, mode):
        plan = _plan(repo)
        selectors = ["--files", "src/b.py", "--files", "src/a.py"] if mode == CODE else []
        assert _main(mode, plan, "--cascade", "--models", A, B, C,
                     "--scope-policy", "preserve", *selectors) == 0
        review, apply = mode_table.resolve(mode).sequence
        assert phases.words() == [review, apply] * 3, "one review and one apply per stage"
        assert not any("--consolidate" in argv for _word, argv in phases.calls)
        files = str(_cascade_dir(plan, mode) / ic.FILES_NAME)
        for stage, (model, argv) in enumerate(zip((A, B, C), phases.reviews()), start=1):
            expected = ["--yes", "--scope-policy", "preserve", "--models", model,
                        "--force", "--rerun-all"]
            if mode == CODE:
                expected += ["--files-from", files]
            assert argv == expected + [HIDDEN, _manifest(plan, mode, stage)], stage
        for argv in phases.applies():
            assert argv == APPLY_FLAGS + ["--scope-policy", "preserve"]
        assert _stage_states(plan, mode) == ["done", "done", "done"]
        assert _record(plan, mode)["state"] == "finished"
        if mode == CODE:
            assert Path(files).read_text(encoding="utf-8") == "src/b.py\nsrc/a.py\n"

    @pytest.mark.parametrize("mode", MODES)
    def test_step_1b_reads_skip_cascade_and_never_runs(self, repo, phases, capsys, mode):
        plan = _plan(repo)
        assert _main(mode, plan, "--cascade", "--models", A, B) == 0
        out = capsys.readouterr().out
        if mode == PLAN:
            assert out.count("[SKIPPED] Step 1b: consolidation: cascade") == 2
        assert not any("--consolidate" in argv for _word, argv in phases.calls)

    def test_the_stages_see_the_ledger_of_the_stages_before(self, repo, phases, capsys):
        plan = _plan(repo)
        assert _main(PLAN, plan, "--cascade", "--models", A, B) == 0
        inputs = _cascade_dir(plan, PLAN) / "stage-2" / "inputs"
        ledger = (inputs / provenance.REVIEWER_LEDGER).read_text(encoding="utf-8")
        assert f"{A} finding" in ledger
        stage_one = _cascade_dir(plan, PLAN) / "stage-1" / "inputs"
        assert (stage_one / provenance.REVIEWER_LEDGER).read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# 2. Resuming a stopped or interrupted cascade
# ---------------------------------------------------------------------------

class TestResume:
    @pytest.mark.parametrize("in_progress", [False, True], ids=["not-started", "batches"])
    def test_an_interruption_after_stage_2s_review_resumes_at_stage_2(
            self, repo, phases, capsys, in_progress):
        plan = _plan(repo)
        phases.kill = ("apply-suggestions", B, "started") if in_progress else (
            "review-plan", B, "after")
        with pytest.raises(Killed):
            _main(PLAN, plan, "--cascade", "--models", A, B, C)
        assert _stage_states(plan, PLAN) == ["done", "running", "pending"]
        phases.kill = None
        phases.calls.clear()
        assert _main(PLAN, plan, "--cascade", "--resume") == 0
        assert phases.words() == ["apply-suggestions", "review-plan", "apply-suggestions"]
        assert all(A not in argv for _word, argv in phases.calls), "no stage-1 argv"
        resumed_apply, stage_3_review, _apply = (argv for _word, argv in phases.calls)
        assert resumed_apply == APPLY_FLAGS + (["--resume"] if in_progress else [])
        assert stage_3_review == ["--yes", "--models", C, "--force", "--rerun-all",
                                  HIDDEN, _manifest(plan, PLAN, 3)]
        assert _stage_states(plan, PLAN) == ["done", "done", "done"]

    def test_a_review_interrupted_mid_fan_out_resumes_without_a_new_round(
            self, repo, phases, capsys):
        plan = _plan(repo)
        phases.kill = ("review-plan", B, "mid")
        with pytest.raises(Killed):
            _main(PLAN, plan, "--cascade", "--models", A, B)
        phases.kill = None
        phases.calls.clear()
        assert _main(PLAN, plan) == 0, "a bare re-run continues the unfinished cascade"
        assert phases.reviews() == [["--yes", "--models", B, HIDDEN, _manifest(plan, PLAN, 2)]], (
            "--force --rerun-all only at a stage's start")
        assert "[IMPROVE_CASCADE] continuing stage 2 of 2" in capsys.readouterr().out

    def test_a_failing_apply_in_stage_2_stops_at_row_14_with_a_cascade_continuation(
            self, repo, phases, capsys):
        plan = _plan(repo)
        phases.fail[("apply-suggestions", B)] = 1
        args = iw.parse_args(PLAN, ["--plan-file", str(plan), "--cascade", "--force",
                                    "--models", A, B, C])
        workflow = iw.ImproveWorkflow(args, PLAN)
        assert workflow.run() == 3
        err = capsys.readouterr().err
        resume = _continuation(PLAN, plan, "--models", A, B, C, "--cascade", "--resume")
        assert "ERROR: Step 2: apply-suggestions did not complete (exited 1)" in err
        assert f"         {resume}" in err
        assert err.count("ERROR: Step 2: apply-suggestions did not complete") == 1
        assert {(r.stage, r.phase) for r in workflow.reports} == {
            (1, "review-plan"), (1, ""), (1, "apply-suggestions"),
            (2, "review-plan"), (2, ""), (2, "apply-suggestions"),
        }
        progress = StateManager(plan).get("improve_plan")
        assert progress["failed_phase"] == "apply-suggestions"
        assert progress["steps"]["stage-1/apply-suggestions"] == "done"
        assert progress["steps"]["stage-2/apply-suggestions"] == "failed"
        assert _stage_states(plan, PLAN) == ["done", "running", "pending"]
        # The continuation picks stage 2's apply up; stage 1 never runs again.
        del phases.fail[("apply-suggestions", B)]
        phases.calls.clear()
        assert _main(PLAN, plan, "--models", A, B, C, "--cascade", "--resume") == 0
        assert phases.words() == ["apply-suggestions", "review-plan", "apply-suggestions"]
        assert phases.reviews()[0][:3] == ["--yes", "--models", C]

    def test_a_bare_continuation_of_an_unfinished_cascade_carries_cascade(
            self, repo, phases, capsys):
        plan = _plan(repo)
        phases.fail[("apply-suggestions", A)] = 1
        assert _main(PLAN, plan, "--cascade", "--models", A, B) == 3
        capsys.readouterr()
        phases.calls.clear()
        assert _main(PLAN, plan) == 3, "the bare re-run continues the cascade, and fails again"
        err = capsys.readouterr().err
        assert f"         {_continuation(PLAN, plan, '--cascade', '--resume')}" in err
        assert phases.words() == ["apply-suggestions"]


# ---------------------------------------------------------------------------
# 3. Zero-result stages and the one result line
# ---------------------------------------------------------------------------

class TestZeroResultStagesAndTheResultLine:
    @pytest.mark.parametrize("mode", MODES)
    def test_a_completed_cascade_ends_on_one_done_line(
            self, repo, phases, capsys, dispatcher_env, mode):
        plan = _plan(repo)
        assert multi_llm.main([mode, str(plan), "--cascade", "--models", A, B]) == 0
        out = capsys.readouterr().out
        result = _one_result(out)
        assert (result["state"], result["exit"], result["next"]) == ("done", 0, None)
        assert "error" not in result
        summary = str(_cascade_dir(plan, mode) / ic.SUMMARY_NAME)
        assert summary in result["artifacts"]

    @pytest.mark.parametrize("mode", MODES)
    def test_a_middle_stage_with_no_result_is_skipped_and_the_cascade_completes(
            self, repo, phases, capsys, dispatcher_env, mode):
        plan = _plan(repo)
        phases.zero[B] = "rate limited"
        assert multi_llm.main([mode, str(plan), "--cascade", "--models", A, B, C]) == 0
        captured = capsys.readouterr()
        result = _one_result(captured.out)
        assert (result["state"], result["exit"], result["next"]) == ("done", 0, None)
        assert "error" not in result
        assert f"[CASCADE_STAGE_SKIPPED] 2 {B}: rate limited" in captured.out
        assert f"[CASCADE_STAGE_SKIPPED] 2 {B}: rate limited" in result["notice"]
        assert _stage_states(plan, mode) == ["done", "skipped", "done"]
        review, apply = mode_table.resolve(mode).sequence
        assert phases.words() == [review, apply, review, review, apply]
        assert phases.reviews()[2][:5] == ["--yes", "--models", C, "--force", "--rerun-all"]
        assert "did not complete" not in captured.err

    def test_the_last_stage_with_no_result_stops_and_its_rerun_starts_it_fresh(
            self, repo, phases, capsys, dispatcher_env):
        plan = _plan(repo)
        phases.zero[B] = "quota"
        assert multi_llm.main([PLAN, str(plan), "--cascade", "--models", A, B]) == 3
        captured = capsys.readouterr()
        result = _one_result(captured.out)
        resume = _continuation(PLAN, plan, "--models", A, B, "--cascade", "--resume")
        assert (result["state"], result["exit"]) == ("needs-human", 3)
        assert result["next"] == resume
        assert "Step 1: review-plan did not complete (exited 3)" in result["error"]
        assert captured.err.count("did not complete") == 1
        assert _stage_states(plan, PLAN) == ["done", "running"]
        assert _record(plan, PLAN)["state"] == "running"
        assert "### Cascade" in captured.out
        del phases.zero[B]
        phases.calls.clear()
        assert multi_llm.main([PLAN, str(plan), "--models", A, B, "--cascade",
                               "--resume"]) == 0
        assert phases.reviews() == [["--yes", "--models", B, "--force", "--rerun-all",
                                     HIDDEN, _manifest(plan, PLAN, 2)]]
        assert _record(plan, PLAN)["state"] == "finished"

    @pytest.mark.parametrize("mode", MODES)
    def test_the_printed_continuation_after_the_finish_is_already_complete(
            self, repo, phases, capsys, dispatcher_env, monkeypatch, mode):
        """A bare re-run of a stopped cascade prints ``--cascade --resume``
        with no model flag; typed after the cascade finished it is exit 2
        ``[ALREADY_COMPLETE]``, never a new cascade of today's defaults (here
        another list, C)."""
        monkeypatch.setattr(ic, "resolve_stage_models",
                            lambda plan, m, args: (list(args.models or [C]), None))
        plan = _plan(repo)
        apply = mode_table.resolve(mode).sequence[1]
        phases.fail[(apply, B)] = 1
        assert multi_llm.main([mode, str(plan), "--cascade", "--models", A, B]) == 3
        capsys.readouterr()
        del phases.fail[(apply, B)]
        assert multi_llm.main([mode, str(plan)]) == 0, "the bare re-run finishes the cascade"
        assert _record(plan, mode)["state"] == "finished"
        capsys.readouterr()
        continuation = iw.continuation_command(
            mode, plan, iw.parse_args(mode, ["--plan-file", str(plan), "--cascade"]))
        assert continuation == _continuation(mode, plan, "--cascade", "--resume")
        phases.calls.clear()
        before = _tree(repo)
        assert multi_llm.main([mode, str(plan), "--cascade", "--resume"]) == 2
        out = capsys.readouterr().out
        result = _one_result(out)
        assert (result["state"], result["exit"]) == ("nothing-to-do", 2)
        assert result["notice"].startswith("[ALREADY_COMPLETE] cascade finished at ")
        assert phases.calls == []
        assert len(list((_out(plan) / mode / ic.CASCADES_DIR).iterdir())) == 1
        assert _tree(repo) == before, "nothing written"

    def test_a_stage_still_open_after_its_steps_is_never_run_again(
            self, repo, phases, capsys, dispatcher_env, monkeypatch):
        """Every step of stage 1 reported done, yet the cascade does not find
        the stage done: the continue call names the stage it follows, so the
        loop stops at exit 3 with the cascade's continuation instead of
        handing stage 1 out again (and again)."""
        plan = _plan(repo)
        monkeypatch.setattr(ic, "_stage_done", lambda *a, **k: False)
        assert multi_llm.main([PLAN, str(plan), "--cascade", "--models", A, B]) == 3
        captured = capsys.readouterr()
        assert phases.words() == ["review-plan", "apply-suggestions"], "stage 1 ran once"
        result = _one_result(captured.out)
        resume = _continuation(PLAN, plan, "--models", A, B, "--cascade", "--resume")
        assert (result["state"], result["exit"], result["next"]) == ("needs-human", 3, resume)
        assert f"stage 1 of 2 ({A}) did not finish" in captured.err
        assert f"         {resume}" in captured.err
        assert _stage_states(plan, PLAN) == ["running", "pending"]
        progress = StateManager(plan).get("improve_plan")
        assert progress["failed_phase"] == "cascade"

    def test_after_a_finished_cascade_a_plain_run_is_already_complete(
            self, repo, phases, capsys):
        plan = _plan(repo)
        assert _main(PLAN, plan, "--cascade", "--models", A) == 0
        phases.calls.clear()
        assert _main(PLAN, plan) == 2
        assert "[ALREADY_COMPLETE] improve-plan:" in capsys.readouterr().out
        assert phases.calls == []


# ---------------------------------------------------------------------------
# 4. A finished cascade typed again, the report and the checkpoint
# ---------------------------------------------------------------------------

class TestFinishedReportAndCheckpoint:
    @pytest.mark.parametrize("route", ["entry", "dispatcher-detach", "worker-backstop"])
    def test_a_finished_cascades_command_is_exit_2_and_launches_nothing(
            self, repo, phases, capsys, dispatcher_env, monkeypatch, route):
        plan = _plan(repo)
        flags = ["--cascade", "--models", A, B]
        assert _main(PLAN, plan, *flags) == 0
        capsys.readouterr()
        _refuse_the_boundary(monkeypatch)
        phases.calls.clear()
        before = _tree(repo)
        if route == "entry":
            assert _main(PLAN, plan, *flags) == 2
        elif route == "dispatcher-detach":
            assert multi_llm.main([PLAN, str(plan), *flags, "--detach"]) == 2
        else:
            args = iw.parse_args(PLAN, ["--plan-file", str(plan), *flags])
            assert iw.ImproveWorkflow(args, PLAN).run() == 2
        out = capsys.readouterr().out
        assert "[ALREADY_COMPLETE] cascade finished at " in out
        assert "### Cascade" in out
        assert phases.calls == []
        assert _tree(repo) == before, "nothing written"
        if route == "dispatcher-detach":
            result = _one_result(out)
            assert (result["state"], result["exit"]) == ("nothing-to-do", 2)
            assert result["notice"].startswith("[ALREADY_COMPLETE] cascade finished at ")

    @pytest.mark.parametrize("mode", MODES)
    def test_the_report_holds_the_table_as_written_and_the_last_stages_steps(
            self, repo, phases, capsys, mode):
        plan = _plan(repo)
        assert _main(mode, plan, "--cascade", "--models", A, B, C) == 0
        out = capsys.readouterr().out
        cascade_dir = _cascade_dir(plan, mode)
        report = out.split(f"## {mode.replace('-', ' ').title()} Summary", 1)[1]
        summary = (cascade_dir / ic.SUMMARY_NAME).read_text(encoding="utf-8")
        table = "\n".join(summary.splitlines()[2:]).strip()
        assert "### Cascade\n\n" + table in report
        assert f"### Stage 3 of 3: {C}" in report
        review = mode_table.resolve(mode).sequence[0]
        assert report.count(f"- Step 1: {review}: done") == 1, "the last stage's steps only"
        files = report.split("### Output files", 1)[1]
        for path in [cascade_dir / ic.SUMMARY_NAME, cascade_dir / "ledger.md",
                     *(cascade_dir / f"stage-{k}" for k in (1, 2, 3))]:
            assert f"- {path}" in files

    def test_the_run_dirs_checkpoint_names_the_stage_while_it_runs(
            self, repo, phases, capsys):
        plan = _plan(repo)
        run_dir = _out(plan) / PLAN
        run_dir.mkdir(parents=True)
        status_file.write_status(str(run_dir), {"phase": PLAN, "state": "running"})
        assert _main(PLAN, plan, "--cascade", "--models", A, B) == 0
        assert phases.stages_seen == [
            {"index": 1, "of": 2, "model": A}, {"index": 1, "of": 2, "model": A},
            {"index": 2, "of": 2, "model": B}, {"index": 2, "of": 2, "model": B},
        ]
        assert status_file.read_status(str(run_dir))["cascade_stage"] is None

    def test_a_foreground_run_with_no_checkpoint_writes_none(self, repo, phases, capsys):
        plan = _plan(repo)
        assert _main(PLAN, plan, "--cascade", "--models", A) == 0
        assert phases.stages_seen == [None, None]
        assert not os.path.exists(status_file.status_path(str(_out(plan) / PLAN)))


# ---------------------------------------------------------------------------
# 5. A refusal between two stages
# ---------------------------------------------------------------------------

class TestARefusalBetweenStages:
    def test_it_ends_on_the_cascade_continuation_never_a_phases_follow_up(
            self, repo, phases, capsys, dispatcher_env, monkeypatch):
        """An archive refusal at stage 1's close-out, after the apply left a
        follow-up of its own on the result line: the line's ``next`` is the
        cascade's continuation, the progress record names the failure."""
        plan = _plan(repo)
        real_apply = phases.apply

        def apply(workflow):
            code = real_apply(workflow)
            cli_common.record_next("STALE: the apply phase's own follow-up")
            return code

        phases.apply = apply

        def refuse(**_kwargs):
            raise cascade_archive.ArchiveRefusal("ERROR: the cascade archive refused (test)")

        monkeypatch.setattr(cascade_archive, "close_out", refuse)
        assert multi_llm.main([PLAN, str(plan), "--cascade", "--models", A, B]) == 3
        captured = capsys.readouterr()
        result = _one_result(captured.out)
        resume = _continuation(PLAN, plan, "--models", A, B, "--cascade", "--resume")
        assert (result["state"], result["exit"], result["next"]) == ("needs-human", 3, resume)
        assert result["error"] == "ERROR: the cascade archive refused (test)"
        assert "ERROR: the cascade archive refused (test)" in captured.err
        assert f"         {resume}" in captured.err
        assert "STALE" not in json.dumps(result)
        progress = StateManager(plan).get("improve_plan")
        assert (progress["current_phase"], progress["failed_phase"]) == ("cascade", "cascade")
        assert progress["steps"]["stage-1/apply-suggestions"] == "done"
        assert _stage_states(plan, PLAN) == ["running", "pending"]

    def test_a_refusal_of_the_first_advance_names_no_continuation(
            self, repo, phases, capsys, dispatcher_env):
        plan = _plan(repo)
        phases.fail[("apply-suggestions", A)] = 1
        assert multi_llm.main([PLAN, str(plan), "--cascade", "--models", A, B]) == 3
        capsys.readouterr()
        phases.calls.clear()
        assert multi_llm.main([PLAN, str(plan), "--models", B]) == 3
        captured = capsys.readouterr()
        result = _one_result(captured.out)
        assert result["next"] is None
        assert captured.err.startswith(f"{ic.CONFLICT_MARKER} --models")
        assert "Once that is dealt with" not in captured.err
        assert phases.calls == []


# ---------------------------------------------------------------------------
# 6. --force without --cascade over an unfinished cascade
# ---------------------------------------------------------------------------

class TestForceWithoutCascade:
    def test_it_abandons_the_cascade_and_runs_todays_single_round(
            self, repo, phases, capsys):
        plan = _plan(repo)
        phases.fail[("apply-suggestions", A)] = 1
        assert _main(PLAN, plan, "--cascade", "--models", A, B) == 3
        record_path = _cascade_dir(plan, PLAN) / ic.RECORD_NAME
        del phases.fail[("apply-suggestions", A)]
        phases.calls.clear()
        capsys.readouterr()
        assert _main(PLAN, plan, "--force", "--models", A, B) == 0
        out = capsys.readouterr().out
        assert "NOTE: --force without --cascade abandons the unfinished cascade" in out
        assert phases.reviews() == [["--yes", "--models", A, B, "--force", "--rerun-all"]]
        assert ic.load_current(plan, PLAN) is None
        assert json.loads(record_path.read_text(encoding="utf-8"))["state"] == "abandoned"
        assert "### Cascade" not in out

    @pytest.mark.parametrize("route", ["entry", "dispatcher-detach"])
    def test_then_the_stale_cascade_continuation_is_refused_before_the_boundary(
            self, repo, phases, capsys, dispatcher_env, monkeypatch, route):
        """The continuation the stopped cascade printed, typed after ``--force``
        alone abandoned it: exit 3 ``[CASCADE_CONFLICT]``, nothing launched,
        nothing written, never a new cascade of today's defaults (C here)."""
        monkeypatch.setattr(ic, "resolve_stage_models",
                            lambda plan, m, args: (list(args.models or [C]), None))
        plan = _plan(repo)
        phases.fail[("apply-suggestions", A)] = 1
        assert _main(PLAN, plan, "--cascade", "--models", A, B) == 3
        del phases.fail[("apply-suggestions", A)]
        assert _main(PLAN, plan, "--force", "--models", A, B) == 0
        assert ic.load_current(plan, PLAN) is None
        capsys.readouterr()
        _refuse_the_boundary(monkeypatch)
        phases.calls.clear()
        cascades = sorted((_out(plan) / PLAN / ic.CASCADES_DIR).iterdir())
        before = _tree(repo)
        if route == "entry":
            assert _main(PLAN, plan, "--cascade", "--resume") == 3
        else:
            assert multi_llm.main([PLAN, str(plan), "--cascade", "--resume", "--detach"]) == 3
        captured = capsys.readouterr()
        assert (f"[CASCADE_CONFLICT] --resume: there is no unfinished improve-plan cascade "
                "to resume (") in captured.err
        assert "--cascade without --resume, or with --models or --quick, starts one." in (
            captured.err)
        assert phases.calls == []
        assert sorted((_out(plan) / PLAN / ic.CASCADES_DIR).iterdir()) == cascades
        assert _tree(repo) == before, "nothing written"
        if route == "dispatcher-detach":
            result = _one_result(captured.out)
            assert (result["state"], result["exit"]) == ("needs-human", 3)

