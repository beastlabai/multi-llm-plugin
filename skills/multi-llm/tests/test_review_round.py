"""A review round's identity (``utils/review_round.py``).

A fan-out stamps a random ``round_id`` into ``.status.json`` before it
replaces anything, and the round's task file, ``validation.json`` and
``report.md`` record it. review-code's partial-completion guard completes the
phase on files only when they record the round on record, and
``util resume-state`` reads the ids: the round's own files give a precise
answer, another round's are stale, files from before round ids keep the
conservative one.

The rounds here are real: the orchestrators run as subprocesses against the
mock provider binaries (``tests/harness``), with validation answered by the
harness's simulated subagents. Nothing reaches a provider.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import SkillRunner  # noqa: E402
from utils import report_markdown, resume_state, review_round  # noqa: E402
from utils.report_parser import (  # noqa: E402
    parse_skipped_issues,
    parse_validation_overrides_issues,
)
from utils.state_manager import StateManager  # noqa: E402
from utils.subagent_tasks import batch_manifest, read_task_file  # noqa: E402
from utils.validation import (  # noqa: E402
    load_validation_results,
    prepare_batched_validation_tasks,
    save_validation_results,
    validation_source_decision,
)

MODEL = "cursor-agent:auto"
MODELS = ["codex:gpt-6-astra:xhigh", "pi:openrouter/deepseek/deepseek-v4.1-flash"]

PLAN_BODY = """# Upload Plan

## Overview
Upload the user's files to the archive.

## Step 1
Retry a failed upload.
"""

TASKS_BODY = (
    "# Implementation Tasks\n\n"
    "## T001: Add the upload client\n"
    "Create the client that sends one file.\n\n"
    "## T002: Retry failed uploads\n"
    "Wrap the client in a bounded retry.\n"
)

SUGGESTION = {
    "title": "Bound the upload retry",
    "desc": "Step 1 retries a failed upload without saying how often. Name the "
            "number of attempts and the backoff between them.",
    "type": "clarification",
    "reference": "## Step 1",
    "importance": "MEDIUM",
}

CODE_ISSUES = [
    {
        "title": "Unchecked upload status",
        "desc": "The client ignores the HTTP status of the upload response.",
        "importance": "medium",
        "file": "src/upload/client.py",
        "line_range": [10, 12],
        "type": "bug",
        "anchor_text": "def send(path):",
    },
    {
        "title": "Retry loop has no bound",
        "desc": "The retry loop never stops on a permanent failure.",
        "importance": "high",
        "file": "src/upload/retry.py",
        "line_range": [4, 9],
        "type": "bug",
        "anchor_text": "while True:",
    },
]

#: (orchestrator, phase dir and resume phase, what every model answers).
PHASES = [
    pytest.param("review_plan", "review-plan", [SUGGESTION], id="review-plan"),
    pytest.param("review_tasks", "review-tasks", [SUGGESTION], id="review-tasks"),
    pytest.param("code_review", "code-review", CODE_ISSUES, id="review-code"),
]
PHASE_NAMES = ["review-plan", "review-tasks", "code-review"]

R1, R2 = "1111222233334444", "aaaabbbbccccdddd"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "upload-plan.md"
    plan.write_text(PLAN_BODY, encoding="utf-8")
    tasks = tmp_path / "upload-plan" / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    (tasks / "tasks.md").write_text(TASKS_BODY, encoding="utf-8")
    return plan


def _implemented(plan: Path) -> None:
    """What review-code needs from the earlier phases: a recorded implement
    and the files it touched, so no scope falls back to git."""
    state = StateManager(plan)
    state.mark_phase_completed("review-plan")
    state.mark_phase_completed("implement")
    state.set("head_at_start", "abc123def")
    for issue in CODE_ISSUES:
        state.track_file(issue["file"], "created", "T001")
    state.save()
    implement = plan.parent / plan.stem / "implement"
    implement.mkdir(parents=True, exist_ok=True)
    (implement / "summary.json").write_text(
        json.dumps({"status": "completed", "tasks_completed": 2, "tasks_total": 2}),
        encoding="utf-8",
    )


def _phase(plan: Path, phase: str) -> Path:
    return plan.parent / plan.stem / phase


def _completed(plan: Path) -> Dict[str, Any]:
    document = json.loads((plan.parent / plan.stem / "state.json").read_text(encoding="utf-8"))
    return document.get("phases_completed") or {}


def _status(phase_dir: Path) -> Dict[str, Any]:
    return json.loads((phase_dir / ".status.json").read_text(encoding="utf-8"))


def _ok(result) -> None:
    assert result.exit_code == 0, (
        f"exit {result.exit_code}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _run(runner: SkillRunner, orchestrator: str, plan: Path, *extra: str):
    return runner.run_orchestrator(
        orchestrator, plan, "--models", MODEL, "--yes", *extra, timeout=120,
    )


def _complete_round(tmp_path: Path, orchestrator: str, phase: str, response):
    """A finished round: the fan-out stops at its validation marker, the
    simulated host answers every batch and ``--reaggregate`` completes it."""
    runner = SkillRunner(tmp_path / "harness")
    runner.set_config({"response": response})
    plan = _plan(tmp_path)
    if orchestrator == "code_review":
        _implemented(plan)
    _ok(_run(runner, orchestrator, plan))
    phase_dir = _phase(plan, phase)
    assert (phase_dir / "validation_tasks.json").is_file()
    runner.simulate_subagents(phase_dir / "validation_tasks.json")
    _ok(runner.run_orchestrator(orchestrator, plan, "--reaggregate", "--yes", timeout=120))
    assert phase in _completed(plan)
    return runner, plan, phase_dir


def _interrupted_force_run(runner: SkillRunner, orchestrator: str, plan: Path):
    """A ``--force`` re-run that stops after its fan-out started and before it
    replaced anything: every reviewer runs again (``--rerun-all``) and fails,
    so it exits before grouping."""
    runner.set_config({"error": "scripted outage", "exit_code": 1})
    result = _run(runner, orchestrator, plan, "--force", "--rerun-all")
    assert result.exit_code == 3, result.stdout
    assert "All models failed" in result.stdout
    return result


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_status(phase_dir: Path, *, round_id: Optional[str], state: str) -> None:
    status: Dict[str, Any] = {"phase": phase_dir.name, "state": state,
                              "models_requested": list(MODELS)}
    if round_id:
        status["round_id"] = round_id
    _write(phase_dir / ".status.json", status)


def _fixture_tasks(phase_dir: Path, batches, round_id: Optional[str]) -> None:
    document: Dict[str, Any] = {"batches": [{"batch_index": index} for index in batches]}
    if round_id:
        document["round_id"] = round_id
    _write(phase_dir / "validation_tasks.json", document)


def _fixture_validation(phase_dir: Path, round_id: Optional[str]) -> None:
    metadata = {"round_id": round_id} if round_id else {}
    _write(phase_dir / "validation.json", {"groups": [], "metadata": metadata})


def _fixture_plan(tmp_path: Path, phase: str):
    plan = tmp_path / "plans" / "feat.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Feat\n\nDo it.\n", encoding="utf-8")
    phase_dir = plan.parent / plan.stem / phase
    phase_dir.mkdir(parents=True)
    return plan, phase_dir


def _classify(plan: Path, phase: str) -> resume_state.ResumeState:
    return resume_state.classify(plan.resolve(), phase)


# ---------------------------------------------------------------------------
# 1. One id per round, in every file the round writes
# ---------------------------------------------------------------------------

class TestARoundRecordsItsIdInEveryFile:
    @pytest.mark.parametrize("orchestrator, phase, response", PHASES)
    def test_status_task_file_validation_and_report_share_one_id(
        self, tmp_path, orchestrator, phase, response
    ):
        _runner, plan, phase_dir = _complete_round(tmp_path, orchestrator, phase, response)

        round_id = _status(phase_dir).get("round_id")
        assert review_round.valid_round(round_id), _status(phase_dir)
        rounds = review_round.round_files(phase_dir)
        assert rounds.recorded == {
            "validation_tasks.json": round_id,
            "validation.json": round_id,
            "report.md": round_id,
        }
        assert set(rounds.owners.values()) == {review_round.OWNED}
        text = (phase_dir / "report.md").read_text(encoding="utf-8")
        assert text.splitlines()[0] == report_markdown.ESCAPED_TEXT_MARKER
        mark = report_markdown.report_round_mark(text)
        assert mark.round_id == round_id and mark.validation and mark.body
        # The seal: the whole report, built from the validation.json beside it.
        assert review_round.report_seal(phase_dir) == (round_id, True, True)
        # The report_data.json sidecar gains no key (docs/development/
        # review-results.md, "Report payload"): the id is in neither page.
        for name in ("report_data.json", "report.html"):
            path = phase_dir / name
            if path.is_file():
                assert round_id not in path.read_text(encoding="utf-8"), name
        assert _classify(plan, phase).case == "a"

    @pytest.mark.parametrize("orchestrator, phase, response", PHASES)
    def test_a_new_fan_out_draws_a_new_id(self, tmp_path, orchestrator, phase, response):
        runner, plan, phase_dir = _complete_round(tmp_path, orchestrator, phase, response)
        first = _status(phase_dir)["round_id"]

        _interrupted_force_run(runner, orchestrator, plan)

        second = _status(phase_dir)["round_id"]
        assert review_round.valid_round(second) and second != first
        # The previous round's files are untouched, and read as its own.
        rounds = review_round.round_files(phase_dir)
        assert rounds.stale() == ("validation_tasks.json", "validation.json")
        assert rounds.owner("report.md") == review_round.STALE

    @pytest.mark.parametrize("orchestrator, phase, response", PHASES)
    def test_resume_state_calls_the_interrupted_round_stale(
        self, tmp_path, orchestrator, phase, response
    ):
        """U62b answered ``uncertain`` here from the checkpoint word alone; the
        ids prove the files another round's, and the answer stays a re-run."""
        runner, plan, phase_dir = _complete_round(tmp_path, orchestrator, phase, response)
        _interrupted_force_run(runner, orchestrator, plan)

        answer = _classify(plan, phase)

        assert (answer.case, answer.action, answer.flags, answer.ownership) == (
            "c", "rerun", ("--force",), "stale")
        assert answer.models == (MODEL,)
        assert "earlier review round" in answer.note


# ---------------------------------------------------------------------------
# 2. Codex's repro: review-code's secondary guard and an interrupted --force
# ---------------------------------------------------------------------------

class TestAnInterruptedForcedReviewIsNotCompletedByAPlainOne:
    def test_the_plain_review_refuses_instead_of_auto_healing(self, tmp_path):
        """A forced review clears the completion and starts its fan-out, then
        stops before replacing the previous round's task file, validation.json
        and report.md. A later plain review used to find all three, record the
        phase complete, write ``state: complete`` and exit 0, so the forced
        review was never finished."""
        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        _interrupted_force_run(runner, "code_review", plan)
        assert "code-review" not in _completed(plan)

        runner.set_config({"response": CODE_ISSUES})
        result = _run(runner, "code_review", plan)

        assert result.exit_code == 3, result.stdout
        assert "Auto-healed" not in result.stdout
        assert "Partial completion detected" in result.stdout
        assert "may not be this round's" in result.stdout
        assert "--force" in result.stdout
        assert "code-review" not in _completed(plan)
        assert _status(phase_dir)["state"] != "complete"

    def test_the_force_run_the_guard_names_finishes_the_round(self, tmp_path):
        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        _interrupted_force_run(runner, "code_review", plan)
        runner.set_config({"response": CODE_ISSUES})

        _ok(_run(runner, "code_review", plan, "--force"))
        runner.simulate_subagents(phase_dir / "validation_tasks.json")
        _ok(runner.run_orchestrator("code_review", plan, "--reaggregate", "--yes", timeout=120))

        assert "code-review" in _completed(plan)
        assert set(review_round.round_files(phase_dir).owners.values()) == {
            review_round.OWNED}

    def test_this_round_s_files_still_auto_heal(self, tmp_path):
        """The heal the guard exists for: this round's verdicts and report are
        on disk and only the completion record is missing."""
        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        state = StateManager(plan)
        state.state["phases_completed"].pop("code-review")
        state.save()

        result = _run(runner, "code_review", plan)

        _ok(result)
        assert "Auto-healed" in result.stdout
        assert "code-review" in _completed(plan)

    def test_an_earlier_round_s_report_is_rebuilt_not_trusted(self, tmp_path):
        """This round's task file and verdicts beside a report.md of another
        round (a stop between the merge and the report): no completion from
        the files, and the way forward is the reaggregate that rebuilds it."""
        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        report = phase_dir / "report.md"
        lines = report.read_text(encoding="utf-8").split("\n")
        lines[1] = report_markdown.round_line(R1)
        report.write_text("\n".join(lines), encoding="utf-8")
        state = StateManager(plan)
        state.state["phases_completed"].pop("code-review")
        state.save()

        result = _run(runner, "code_review", plan)

        assert result.exit_code == 3, result.stdout
        assert "Auto-healed" not in result.stdout
        assert "--reaggregate" in result.stdout
        assert "code-review" not in _completed(plan)

    def test_a_stop_after_the_regroup_is_not_resumed_from_the_old_task_file(
        self, tmp_path
    ):
        """Past the regroup the previous round's validation.json is retired but
        its task file and batch files stay until the new task file lands. The
        guard used to answer ``--reaggregate``, which could merge them."""
        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        (phase_dir / "validation.json").rename(phase_dir / "validation.retired.json")
        status = _status(phase_dir)
        status.update(state="grouping_complete", round_id=R2)
        _write(phase_dir / ".status.json", status)
        state = StateManager(plan)
        state.state["phases_completed"].pop("code-review")
        state.save()

        result = _run(runner, "code_review", plan)

        assert result.exit_code == 3, result.stdout
        assert "--reaggregate" not in result.stdout
        assert "--force" in result.stdout
        assert f"records round {R2}" in result.stdout


# ---------------------------------------------------------------------------
# 3. util resume-state: owned, stale and legacy files, every phase
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phase", PHASE_NAMES)
class TestResumeStateReadsTheRoundIds:
    def test_an_owned_task_file_names_the_missing_batches(self, tmp_path, phase):
        """The checkpoint word no longer matters once the id ties the task file
        to the round: U62b answered ``rerun --force ownership=uncertain`` for
        this dir (its state is not ``validation_pending``)."""
        plan, phase_dir = _fixture_plan(tmp_path, phase)
        _fixture_status(phase_dir, round_id=R1, state="grouping_complete")
        _fixture_tasks(phase_dir, [1, 2], R1)
        _write(phase_dir / "validation_batch_1.json", [])

        answer = _classify(plan, phase)

        assert (answer.case, answer.action, answer.batches, answer.ownership) == (
            "b", "validate", (2,), "-")

    def test_an_owned_whole_round_reaggregates(self, tmp_path, phase):
        plan, phase_dir = _fixture_plan(tmp_path, phase)
        _fixture_status(phase_dir, round_id=R1, state="complete")
        _fixture_tasks(phase_dir, [1, 2], R1)
        for index in (1, 2):
            _write(phase_dir / f"validation_batch_{index}.json", [])
        _fixture_validation(phase_dir, R1)

        assert _classify(plan, phase).line() == (
            "[RESUME_STATE] case=b action=reaggregate models=- batches=- flags=- ownership=-")

    def test_an_owned_task_file_with_no_batch_yet_validates_all(self, tmp_path, phase):
        plan, phase_dir = _fixture_plan(tmp_path, phase)
        _fixture_status(phase_dir, round_id=R1, state="grouping_complete")
        _fixture_tasks(phase_dir, [0, 1], R1)

        answer = _classify(plan, phase)

        assert (answer.case, answer.action, answer.batches) == ("f", "validate", (0, 1))

    @pytest.mark.parametrize("state", ["validation_pending", "models_running", "complete"])
    def test_another_round_s_task_file_is_stale_whatever_the_checkpoint_says(
        self, tmp_path, phase, state
    ):
        plan, phase_dir = _fixture_plan(tmp_path, phase)
        _fixture_status(phase_dir, round_id=R2, state=state)
        _fixture_tasks(phase_dir, [1], R1)
        _write(phase_dir / "validation_batch_1.json", [])
        _fixture_validation(phase_dir, R1)

        answer = _classify(plan, phase)

        assert answer.line() == (
            f"[RESUME_STATE] case=c action=rerun models={','.join(MODELS)} "
            "batches=- flags=--force ownership=stale")
        assert f"records round {R1}" in answer.note and R2 in answer.note

    def test_another_round_s_validation_beside_this_round_s_task_file(self, tmp_path, phase):
        plan, phase_dir = _fixture_plan(tmp_path, phase)
        _fixture_status(phase_dir, round_id=R2, state="validation_pending")
        _fixture_tasks(phase_dir, [1], R2)
        _write(phase_dir / "validation_batch_1.json", [])
        _fixture_validation(phase_dir, R1)

        answer = _classify(plan, phase)

        assert (answer.case, answer.action, answer.ownership) == ("c", "rerun", "stale")

    def test_only_the_report_of_another_round_still_reaggregates(self, tmp_path, phase):
        """The reaggregate rebuilds a report; nothing reads the old one."""
        plan, phase_dir = _fixture_plan(tmp_path, phase)
        _fixture_status(phase_dir, round_id=R2, state="validation_pending")
        _fixture_tasks(phase_dir, [1], R2)
        _write(phase_dir / "validation_batch_1.json", [])
        _fixture_validation(phase_dir, R2)
        (phase_dir / "report.md").write_text(
            f"{report_markdown.ESCAPED_TEXT_MARKER}\n{report_markdown.round_line(R1)}\n# Old\n",
            encoding="utf-8")

        assert _classify(plan, phase).action == "reaggregate"

    def test_a_legacy_task_file_keeps_the_checkpoint_word_proof(self, tmp_path, phase):
        """No id on the task file: U62b's answer, whichever side lacks one."""
        plan, phase_dir = _fixture_plan(tmp_path, phase)
        _fixture_status(phase_dir, round_id=R2, state="validation_pending")
        _fixture_tasks(phase_dir, [1, 2], None)
        _write(phase_dir / "validation_batch_1.json", [])

        answer = _classify(plan, phase)

        assert (answer.case, answer.action, answer.batches) == ("b", "validate", (2,))

    @pytest.mark.parametrize("status_round, tasks_round", [(R2, None), (None, R1), (None, None)])
    def test_a_legacy_file_without_the_checkpoint_word_stays_uncertain(
        self, tmp_path, phase, status_round, tasks_round
    ):
        plan, phase_dir = _fixture_plan(tmp_path, phase)
        _fixture_status(phase_dir, round_id=status_round, state="models_running")
        _fixture_tasks(phase_dir, [1], tasks_round)
        _fixture_validation(phase_dir, None)

        answer = _classify(plan, phase)

        assert (answer.case, answer.action, answer.flags, answer.ownership) == (
            "c", "rerun", ("--force",), "uncertain")

    def test_a_checkpoint_without_an_id_beside_a_task_file_with_one(self, tmp_path, phase):
        """A checkpoint rewritten by a version without round ids proves
        nothing by id: the checkpoint word decides, as before."""
        plan, phase_dir = _fixture_plan(tmp_path, phase)
        _fixture_status(phase_dir, round_id=None, state="validation_pending")
        _fixture_tasks(phase_dir, [1], R1)

        answer = _classify(plan, phase)

        assert (answer.case, answer.action, answer.batches) == ("f", "validate", (1,))


class TestFilesFromBeforeRoundIds:
    @pytest.mark.parametrize("orchestrator, phase, response", PHASES)
    def test_a_round_without_ids_still_reaggregates_and_completes(
        self, tmp_path, orchestrator, phase, response
    ):
        """Strip every id from a real round, as a version without round ids
        wrote it, and reopen the phase: the reaggregate reads it as before and
        writes no id the checkpoint does not hold."""
        runner, plan, phase_dir = _complete_round(tmp_path, orchestrator, phase, response)
        status = _status(phase_dir)
        del status["round_id"]
        status["state"] = "validation_pending"
        _write(phase_dir / ".status.json", status)
        tasks = json.loads((phase_dir / "validation_tasks.json").read_text(encoding="utf-8"))
        del tasks["round_id"]
        (phase_dir / "validation_tasks.json").write_text(json.dumps(tasks, indent=2),
                                                         encoding="utf-8")
        validation = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        del validation["metadata"]["round_id"]
        _write(phase_dir / "validation.json", validation)
        report = (phase_dir / "report.md").read_text(encoding="utf-8").split("\n")
        del report[1]
        (phase_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
        state = StateManager(plan)
        state.state["phases_completed"].pop(phase)
        state.save()
        assert _classify(plan, phase).action == "reaggregate"

        _ok(runner.run_orchestrator(orchestrator, plan, "--reaggregate", "--yes", timeout=120))

        assert phase in _completed(plan)
        rounds = review_round.round_files(phase_dir)
        assert rounds.current is None
        assert set(rounds.recorded.values()) == {None}


def test_the_ownership_word_is_documented():
    rendered = resume_state.render_cases()
    assert "ownership=stale" in rendered and "ownership=uncertain" in rendered


# ---------------------------------------------------------------------------
# 4. The writers: an added key, and nothing else moves
# ---------------------------------------------------------------------------

def _groups():
    return [
        {"theme": f"Theme {index}", "suggestions": [
            {"type": "clarification", "section": "## Step 1",
             "details": f"Detail {index}", "importance": "MEDIUM",
             "source_model": MODELS[index % 2]}],
         "models": [MODELS[index % 2]], "group_hash": f"{index:016x}"}
        for index in range(6)
    ]


def _materialize(phase_dir: Path, groups, round_id: Optional[str]):
    return prepare_batched_validation_tasks(
        groups=groups, context=PLAN_BODY, output_dir=str(phase_dir),
        plan_file="", model="auto", round_id=round_id,
    )


class TestTheTaskFileDiffersOnlyByTheId:
    @pytest.mark.parametrize("groups", [_groups(), []], ids=["six-groups", "no-groups"])
    def test_strip_the_id_and_the_bytes_are_the_same(self, tmp_path, groups):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        without = _materialize(phase_dir, groups, None)
        sidecars = {p.name: p.read_bytes() for p in phase_dir.iterdir()}
        stamped = _materialize(phase_dir, groups, R1)

        assert "round_id" not in without
        assert stamped["round_id"] == R1
        assert list(stamped)[-1] == "round_id"  # appended, nothing reordered
        stripped = dict(stamped)
        del stripped["round_id"]
        assert json.dumps(stripped, indent=2).encode() == json.dumps(without, indent=2).encode()
        # The prompt sidecars and the batch manifest do not see it either.
        assert {p.name: p.read_bytes() for p in phase_dir.iterdir()} == sidecars
        assert batch_manifest(stamped) == batch_manifest(without)

    def test_both_files_load_through_the_task_file_reader(self, tmp_path):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        for round_id in (None, R1):
            path = phase_dir / "validation_tasks.json"
            path.write_text(json.dumps(_materialize(phase_dir, _groups(), round_id),
                                       indent=2), encoding="utf-8")
            document = read_task_file(str(path), phase_dir=str(phase_dir))
            assert review_round.task_document_round(document) == round_id

    def test_an_invalid_id_writes_no_key(self, tmp_path):
        phase_dir = tmp_path / "review-plan"
        phase_dir.mkdir()
        assert "round_id" not in _materialize(phase_dir, _groups(), "not-a-round")


class TestValidationJsonCarriesTheId:
    def test_metadata_gains_the_key_and_readers_ignore_it(self, tmp_path):
        verdicts = [{"group_index": 0, "status": "valid", "reason": "ok", "confidence": 0.9}]
        legacy, stamped = tmp_path / "legacy.json", tmp_path / "stamped.json"
        save_validation_results(verdicts, legacy, model="m")
        save_validation_results(verdicts, stamped, model="m", round_id=R1)

        assert review_round.validation_file_round(legacy) is None
        assert review_round.validation_file_round(stamped) == R1
        old = json.loads(legacy.read_text(encoding="utf-8"))
        new = json.loads(stamped.read_text(encoding="utf-8"))
        del new["metadata"]["round_id"], new["metadata"]["timestamp"]
        del old["metadata"]["timestamp"]
        assert new == old
        assert load_validation_results(stamped) == load_validation_results(legacy)

    @pytest.mark.parametrize("document", [
        [], [{"group_index": 0, "status": "valid"}],
        {"groups": [], "metadata": {"schema_version": "2.1"}},
    ], ids=["empty-list", "v1-list", "no-round"])
    def test_files_from_before_round_ids_still_load(self, tmp_path, document):
        path = tmp_path / "validation.json"
        _write(path, document)
        assert review_round.validation_file_round(path) is None
        assert isinstance(load_validation_results(path), list)
        (tmp_path / "grouped.json").write_text("[]", encoding="utf-8")
        reason, _note = validation_source_decision(path, tmp_path / "grouped.json")
        assert reason is None

    def test_the_round_does_not_change_the_reuse_decision(self, tmp_path):
        verdicts = [{"group_index": 0, "status": "valid", "reason": "ok"}]
        (tmp_path / "grouped.json").write_text("[]", encoding="utf-8")
        decisions = []
        for round_id in (None, R1):
            path = tmp_path / "validation.json"
            save_validation_results(verdicts, path, model="m", batch_sha256={},
                                    batch_manifest=None, round_id=round_id)
            decisions.append(validation_source_decision(path, tmp_path / "grouped.json"))
        assert decisions[0] == decisions[1]


class TestTheReportLine:
    def test_read_only_from_the_line_after_the_marker(self):
        marker, line = report_markdown.ESCAPED_TEXT_MARKER, report_markdown.round_line(R1)
        assert report_markdown.report_round(f"{marker}\n{line}\n# Title\n") == R1
        assert report_markdown.report_round(f"{marker}\r\n{line}\r\n# T\r\n") == R1
        # Model text below cannot claim a round, and no marker means no line.
        assert report_markdown.report_round(f"{marker}\n# Title\n{line}\n") is None
        assert report_markdown.report_round(f"# Title\n{line}\n") is None
        assert report_markdown.report_round("") is None

    def test_a_report_from_before_round_ids_parses_as_before(self, tmp_path):
        body = (
            "# Code Review Report\n\n---\n\n"
            "### 1. Unchecked status [abcdef12]\n- [x] Skip\n\nText.\n"
        )
        decisions = []
        for head in ("", f"{report_markdown.ESCAPED_TEXT_MARKER}\n",
                     f"{report_markdown.ESCAPED_TEXT_MARKER}\n"
                     f"{report_markdown.round_line(R1)}\n"):
            path = tmp_path / "report.md"
            path.write_text(head + body, encoding="utf-8")
            decisions.append((parse_skipped_issues(str(path)),
                              parse_validation_overrides_issues(str(path))))
            assert review_round.report_file_round(path) == (R1 if "round" in head else None)
        assert decisions[0] == decisions[1] == decisions[2]

    def test_the_line_comes_from_the_claim_never_the_checkpoint(self, tmp_path):
        text = f"{report_markdown.ESCAPED_TEXT_MARKER}\n# Report\n\nBody.\n"
        _write(tmp_path / ".status.json", {"round_id": R2})
        # No claim (a direct caller) or a claim with no round: no line.
        assert review_round.report_text(None, text) == text
        assert review_round.report_text(review_round.Claim(str(tmp_path), None), text) == text
        claim = review_round.Claim(str(tmp_path), R1)
        mark = report_markdown.report_round_mark(review_round.report_text(claim, text))
        assert (mark.round_id, mark.validation) == (R1, None) and mark.body

    @pytest.mark.parametrize("recorded", [R2, None])
    def test_verdicts_of_another_or_no_round_get_no_line(self, tmp_path, recorded):
        """The report never claims a round its verdicts did not come from."""
        text = f"{report_markdown.ESCAPED_TEXT_MARKER}\n# Report\n"
        _fixture_validation(tmp_path, recorded)
        claim = review_round.Claim(str(tmp_path), R1)

        verdicts = review_round.verdicts_of((tmp_path / "validation.json").read_bytes())
        assert review_round.report_text(claim, text, verdicts) == text

    def test_the_seal_binds_the_validation_bytes_and_the_whole_report(self, tmp_path):
        _write(tmp_path / ".status.json", {"round_id": R1})
        _fixture_validation(tmp_path, R1)
        claim = review_round.Claim(str(tmp_path), R1)
        text = f"{report_markdown.ESCAPED_TEXT_MARKER}\n# Report\n\n" + "Finding.\n" * 50
        review_round.publish_report(
            tmp_path / "report.md", text, claim=claim,
            verdicts=review_round.verdicts_of((tmp_path / "validation.json").read_bytes()))
        assert review_round.report_seal(tmp_path) == (R1, True, True)

        report = tmp_path / "report.md"
        whole = report.read_text(encoding="utf-8")
        report.write_text(whole.replace("\n", "\r\n"), encoding="utf-8")
        assert review_round.report_seal(tmp_path) == (R1, True, True)  # CRLF-safe
        report.write_text("\n".join(whole.split("\n")[:2]) + "\n", encoding="utf-8")
        assert review_round.report_seal(tmp_path) == (R1, False, True)  # cut short
        report.write_text(whole, encoding="utf-8")
        validation = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))
        validation["groups"] = [{"group_index": 0, "status": "valid"}]
        _write(tmp_path / "validation.json", validation)  # same round, merged again
        assert review_round.report_seal(tmp_path) == (R1, True, False)


class TestTheRoundModule:
    def test_ids_are_random_hex(self):
        ids = {review_round.new_round_id() for _ in range(50)}
        assert len(ids) == 50
        assert all(review_round.valid_round(value) for value in ids)

    def test_start_round_keeps_the_rest_of_the_checkpoint(self, tmp_path):
        _write(tmp_path / ".status.json", {"state": "validation_pending", "round_id": R1,
                                           "models_requested": MODELS})
        claim = review_round.start_round(tmp_path)
        status = _status(tmp_path)
        assert claim.started and claim.round_id != R1
        assert status["round_id"] == claim.round_id
        # The stamp replaces the checkpoint word in the same update (point 2).
        assert status["state"] == "models_running"
        assert status["models_requested"] == MODELS

    @pytest.mark.parametrize("current, recorded, expected", [
        (R1, R1, "owned"), (R1, R2, "stale"), (R1, None, "legacy"),
        (None, R1, "legacy"), (None, None, "legacy"),
    ])
    def test_ownership(self, current, recorded, expected):
        assert review_round.ownership(current, recorded) == expected

    def test_a_malformed_id_is_no_id(self, tmp_path):
        _write(tmp_path / ".status.json", {"round_id": "2026-11-08T10:00:00"})
        _write(tmp_path / "validation_tasks.json", {"round_id": 12})
        rounds = review_round.round_files(tmp_path)
        assert rounds.current is None
        assert rounds.owner("validation_tasks.json") == review_round.LEGACY


# ---------------------------------------------------------------------------
# 5. Codex round 2: each invocation writes for its own round, and only while
#    the checkpoint still records it
# ---------------------------------------------------------------------------

def _wait_for(path: Path, timeout_s: float = 60.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)


def _snapshot(plan: Path, *, skip=()) -> Dict[str, bytes]:
    """Every file under the plan's output dir (state.json and each phase dir)."""
    root = plan.parent / plan.stem
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name not in skip}


def _holder_pid(phase_dir: Path) -> Any:
    from utils import process_control

    return (process_control.read_lock_holder(str(phase_dir / ".lock")) or {}).get("pid")


def _refused_at_the_lock(result, holder_pid) -> None:
    assert result.exit_code == 3, result.stdout + result.stderr
    assert (f"another run of this phase is in progress (pid {holder_pid}"
            in result.stdout), result.stdout + result.stderr
    assert "stopped before it read or changed anything" in result.stdout
    assert "Running code review" not in result.stdout


class TestTwoOverlappingRuns:
    @pytest.mark.parametrize("orchestrator, phase, response", PHASES)
    def test_the_second_run_is_refused_at_the_lock_and_the_first_finishes(
        self, tmp_path, orchestrator, phase, response
    ):
        """Codex's repro: run A is held in its model call after its opening
        checkpoint and a forced run B starts. B used to stamp a round of its
        own; A then wrote B's id into its task file, or stopped at its claim
        check. Now A holds the phase dir's lock for its whole run: B is
        refused at its entry, before it reads or writes anything, and A
        finishes its round untouched."""
        import threading

        runner = SkillRunner(tmp_path / "harness")
        plan = _plan(tmp_path)
        if orchestrator == "code_review":
            _implemented(plan)
        phase_dir = _phase(plan, phase)
        gate = tmp_path / "release-a"
        runner.set_config({"response": response})
        runner.extra_env["MOCK_LLM_HOLD"] = str(gate)
        outcome: Dict[str, Any] = {}
        run_a = threading.Thread(
            target=lambda: outcome.update(a=_run(runner, orchestrator, plan)))
        run_a.start()
        try:
            _wait_for(Path(str(gate) + ".waiting"))
            round_a = _status(phase_dir)["round_id"]
            holder = _holder_pid(phase_dir)
            del runner.extra_env["MOCK_LLM_HOLD"]
            before = _snapshot(plan, skip=(".status.json",))
            run_b = _run(runner, orchestrator, plan, "--force", "--rerun-all")
            after = _snapshot(plan, skip=(".status.json",))
            round_after_b = _status(phase_dir)["round_id"]
        finally:
            gate.write_text("go", encoding="utf-8")
            run_a.join(timeout=180)
        result = outcome["a"]

        assert isinstance(holder, int)
        _refused_at_the_lock(run_b, holder)
        assert after == before
        assert round_after_b == round_a
        _ok(result)
        assert _status(phase_dir)["round_id"] == round_a
        assert review_round.task_file_round(phase_dir / "validation_tasks.json") == round_a
        assert _classify(plan, phase).action == "validate"
        from utils import process_control

        assert process_control.lock_is_free(str(phase_dir / ".lock"))


#: A review-code run held at one of Codex's interleaving points: it writes
#: ``gate`` and waits for ``go`` on stdin, then carries on.
_PAUSED_RUN = r"""
import asyncio, sys
from pathlib import Path
sys.path[:0] = [sys.argv[4]]
import code_review_orchestrator as cro
from utils import review_round as rr

mode, plan, gate = sys.argv[1], sys.argv[2], Path(sys.argv[3])

def pause():
    gate.write_text("ready", encoding="utf-8")
    assert sys.stdin.readline().strip() == "go"

if mode == "opening":
    write = cro.write_status
    def opening(path, data):
        if data.get("state") == "models_running":
            pause()
        return write(path, data)
    cro.write_status = opening
elif mode == "completion":
    check = rr.Claim.check
    def checked(self, doing):
        check(self, doing)
        if doing == "record the phase complete":
            pause()
    rr.Claim.check = checked
elif mode == "grouped":
    retire = cro.retire_validation_for_regroup
    def retiring(*args, **kwargs):
        pause()
        return retire(*args, **kwargs)
    cro.retire_validation_for_regroup = retiring
elif mode == "held":
    replace = rr.replace_file
    def replacing(path, data):
        if str(path).endswith("validation_tasks.json"):
            pause()
        return replace(path, data)
    rr.replace_file = replacing
elif mode == "seal":
    report = cro.generate_review_report
    def reported(*args, **kwargs):
        text = report(*args, **kwargs)
        pause()
        return text
    cro.generate_review_report = reported
flags = ["--reaggregate"] if mode in ("completion", "seal") else ["--force", "--rerun-all"]
code = asyncio.run(cro.main(["--plan-file", plan, "--models", sys.argv[5], "--yes", *flags]))
sys.exit(code or 0)
"""


class TestCodexInterleavingsStopAtTheLock:
    """Codex's round-2 interleavings, each with run A held at the point
    Codex paused it. Each second run used to damage a round: A restored a
    superseded id (opening, point 1), marked B's phase complete (completion,
    point 2), retired B's verdicts (grouped, point 3), published over B's
    files after B's mutex wait expired (held: A paused inside its claim's
    critical section, point 4), or sealed its report to B's verdicts (seal,
    point 5). B is now refused at the phase lock before it writes anything,
    and A's round ends whole."""

    @pytest.mark.parametrize("mode", ["opening", "completion", "grouped", "held", "seal"])
    def test_the_second_run_changes_nothing_and_the_first_ends_whole(self, tmp_path, mode):
        import subprocess
        import time

        if mode in ("completion", "seal"):
            runner, plan, phase_dir = _complete_round(
                tmp_path, "code_review", "code-review", CODE_ISSUES)
        else:
            runner = SkillRunner(tmp_path / "harness")
            runner.set_config({"response": CODE_ISSUES})
            plan = _plan(tmp_path)
            _implemented(plan)
            phase_dir = _phase(plan, "code-review")
        driver = tmp_path / "paused_run.py"
        driver.write_text(_PAUSED_RUN, encoding="utf-8")
        gate = tmp_path / "a-ready"
        log = tmp_path / "a.log"
        with log.open("w", encoding="utf-8") as sink:
            run_a = subprocess.Popen(
                [sys.executable, str(driver), mode, str(plan), str(gate),
                 str(SKILL_ROOT), MODEL],
                stdin=subprocess.PIPE, stdout=sink, stderr=subprocess.STDOUT,
                env=runner._build_env(), cwd=str(SKILL_ROOT), text=True,
            )
            try:
                deadline = time.monotonic() + 90
                while not gate.exists():
                    assert run_a.poll() is None, log.read_text(encoding="utf-8")
                    assert time.monotonic() < deadline, "run A never reached its pause"
                    time.sleep(0.05)
                round_a = review_round.current_round(phase_dir)
                if mode == "seal":
                    # B's verdicts reject every finding A's report keeps.
                    for batch in phase_dir.glob("validation_batch_*.json"):
                        data = json.loads(batch.read_text(encoding="utf-8"))
                        for item in data["groups"]:
                            item["status"], item["reason"] = "invalid", "B rejects it"
                        batch.write_text(json.dumps(data), encoding="utf-8")
                elif mode == "grouped":
                    runner.set_config({"response": [dict(
                        CODE_ISSUES[0], title="Different issue from B",
                        desc="B's round has other findings.")]})
                before = _snapshot(plan)
                run_b = runner.run_orchestrator(
                    "code_review", plan, "--models", MODEL, "--yes",
                    *(["--reaggregate"] if mode == "seal" else ["--force", "--rerun-all"]),
                    timeout=120,
                )
                after = _snapshot(plan)
                run_a.communicate("go\n", timeout=180)
            finally:
                if run_a.poll() is None:
                    run_a.kill()
        output = log.read_text(encoding="utf-8")

        _refused_at_the_lock(run_b, run_a.pid)
        assert after == before
        assert run_a.returncode == 0, output
        assert review_round.current_round(phase_dir) == round_a
        rounds = review_round.round_files(phase_dir)
        assert rounds.recorded["validation_tasks.json"] == round_a
        if mode in ("completion", "seal"):
            assert "code-review" in _completed(plan)
            assert rounds.owners == {name: "owned" for name in rounds.owners}
            assert review_round.report_seal(phase_dir) == (round_a, True, True)


class TestTheClaim:
    def test_a_publication_after_another_stamp_is_refused(self, tmp_path, capsys):
        claim = review_round.start_round(tmp_path)
        target = tmp_path / "validation_tasks.json"
        claim.publish(target, '{"round_id": "%s"}' % claim.round_id, "write it")
        before = target.read_bytes()
        other = review_round.start_round(tmp_path)

        with pytest.raises(review_round.RoundTakenOver) as stop:
            claim.publish(target, "{}", "write validation_tasks.json")

        assert stop.value.code == 3
        assert target.read_bytes() == before
        out = capsys.readouterr().out
        assert claim.round_id in out and other.round_id in out
        assert "write validation_tasks.json" in out
        # The completion decision is refused the same way.
        with pytest.raises(review_round.RoundTakenOver):
            claim.check("record the phase complete")
        other.check("record the phase complete")

    def test_a_reaggregate_claims_the_round_on_record_and_holds_it(self, tmp_path):
        _write(tmp_path / ".status.json", {"round_id": R1})
        claim = review_round.continue_round(tmp_path)
        assert (claim.round_id, claim.started) == (R1, False)
        _write(tmp_path / ".status.json", {"round_id": R2})
        with pytest.raises(review_round.RoundTakenOver):
            claim.check("merge")

    def test_a_round_without_an_id_is_held_until_one_is_stamped(self, tmp_path):
        claim = review_round.continue_round(tmp_path)
        assert claim.round_id is None
        claim.check("merge")
        review_round.start_round(tmp_path)
        with pytest.raises(review_round.RoundTakenOver):
            claim.check("merge")

    def test_a_checkpoint_that_lost_the_id_is_refused(self, tmp_path, capsys):
        """Codex's lost-id probe: B stamped and published, then the checkpoint
        was replaced without an id, and A's stale report went through. A claim
        with an established id now holds only while the checkpoint records
        it: nothing ties a dir whose checkpoint lost it to this run."""
        claim = review_round.start_round(tmp_path)
        report = tmp_path / "report.md"
        claim.publish(report, "this round's report\n", "write report.md")
        _write(tmp_path / ".status.json", {"phase": "code-review", "state": "models_running"})

        with pytest.raises(review_round.RoundTakenOver) as stop:
            claim.publish(report, "a stale report\n", "write report.md")

        assert stop.value.code == 3
        assert report.read_text(encoding="utf-8") == "this round's report\n"
        assert "no longer records that round" in capsys.readouterr().out

    def test_the_opening_write_keeps_the_round_it_did_not_record(self, tmp_path):
        """Checkpoint writes are best effort (test_reaggregate_identity.py,
        "...CheckpointWriteFailed..."): an opening write that landed without
        the id gets it back under the same mutex hold, so the run carries on."""
        from utils.status_file import write_status

        claim = review_round.start_round(tmp_path)
        with claim.opening():
            write_status(str(tmp_path), {"phase": "code-review", "state": "models_running"})
        assert review_round.current_round(tmp_path) == claim.round_id
        claim.check("merge")

    def test_an_opening_write_after_another_stamp_is_refused(self, tmp_path):
        """Codex point 1: only start_round establishes a round; the opening
        write never restores a superseded one."""
        from utils.status_file import write_status

        claim = review_round.start_round(tmp_path)
        other = review_round.start_round(tmp_path)
        with pytest.raises(review_round.RoundTakenOver):
            with claim.opening():
                write_status(str(tmp_path), {"state": "models_running",
                                             "round_id": claim.round_id})
        assert review_round.current_round(tmp_path) == other.round_id

    def test_a_stamp_that_does_not_land_stops_the_run(self, tmp_path, monkeypatch):
        from utils import status_file

        monkeypatch.setattr(status_file, "update_status", lambda *_a, **_k: None)
        with pytest.raises(review_round.RoundTakenOver) as stop:
            review_round.start_round(tmp_path)
        assert stop.value.code == 3

    def test_a_checkpoint_mutex_that_cannot_be_taken_refuses(self, tmp_path, monkeypatch,
                                                              capsys):
        """Codex's timeout probe: A held the checkpoint mutex, B's wait expired
        and B stamped and published unlocked. A stamp or publication whose
        mutex cannot be taken now stops the run instead."""
        import subprocess

        from utils import process_control, status_file

        holder = subprocess.Popen(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, sys.argv[1]);"
             "from utils import process_control as pc;"
             "m = pc.FileMutex(sys.argv[2]); assert m.acquire();"
             "print('held', flush=True); sys.stdin.readline()",
             str(SKILL_ROOT), status_file.status_lock_path(str(tmp_path))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            assert holder.stdout.readline().strip() == "held"
            acquire = process_control.FileMutex.acquire
            monkeypatch.setattr(process_control.FileMutex, "acquire",
                                lambda self, **kw: acquire(self, timeout_s=0.2))
            claim = review_round.Claim(str(tmp_path), R1)
            _write(tmp_path / ".status.json", {"round_id": R1})
            report = tmp_path / "report.md"

            with pytest.raises(review_round.RoundTakenOver) as stop:
                review_round.start_round(tmp_path)
            assert stop.value.code == 3
            assert review_round.current_round(tmp_path) == R1
            with pytest.raises(review_round.RoundTakenOver):
                claim.publish(report, "unlocked\n", "write report.md")
            assert not report.exists()
            assert "never decided unlocked" in capsys.readouterr().out
        finally:
            holder.communicate("\n", timeout=10)

    def test_files_are_replaced_whole(self, tmp_path):
        """A rename, never an in-place rewrite: a reader or a crash sees the old
        file or the new one (report.md and validation.json alike)."""
        report = tmp_path / "report.md"
        report.write_text("old\n", encoding="utf-8")
        inode = report.stat().st_ino
        review_round.publish_report(report, "new\n")
        assert report.read_text(encoding="utf-8") == "new\n"
        assert report.stat().st_ino != inode

        validation = tmp_path / "validation.json"
        validation.write_text("[]\n", encoding="utf-8")
        inode = validation.stat().st_ino
        save_validation_results([], validation, model="m", batch_sha256={},
                                batch_manifest=None, round_id=R1)
        assert validation.stat().st_ino != inode
        assert review_round.validation_file_round(validation) == R1
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "report.md", "validation.json"]


# ---------------------------------------------------------------------------
# 6. One writer per phase dir (utils/detach.py, PhaseRun)
# ---------------------------------------------------------------------------

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="the lock hand-off is POSIX")


def _args(**flags):
    import argparse

    values = {"detach": False, "internal_worker": False, "internal_run_id": None,
              "host": None, "force": False, "if_not_running": False, **flags}
    return argparse.Namespace(**values)


#: A live process holding ``sys.argv[2]`` until its stdin closes. With a run
#: id in ``sys.argv[3]`` it holds it as a detached run would: its label names
#: the run and the checkpoint beside the lock records it.
_HOLDER = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
from utils import process_control as pc, status_file as sf
lock_file, run_id = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
me = {"pid": os.getpid(), "process_start_time": pc.process_start_time(os.getpid())}
lock = pc.PhaseLock(lock_file)
if run_id:
    lock.acquire({"run_id": run_id, **me})
    sf.claim_run(os.path.dirname(lock_file), {
        "run_id": run_id, "state": "running", "detached": True, **me,
        "heartbeat_at": sf.now_iso(), "exit_code": None,
        "log_path": os.path.join(os.path.dirname(lock_file), "orchestrator-run.log"),
    })
else:
    lock.acquire({"holder": "review-code (foreground)", **me})
print("held", flush=True)
sys.stdin.readline()
"""


def _hold_in_another_process(lock: Path, run_id: str = ""):
    """A live process holding *lock* as a run would, until its stdin closes."""
    import subprocess

    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(SKILL_ROOT), str(lock), *([run_id] if run_id else [])],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    assert holder.stdout.readline().strip() == "held"
    return holder


#: A detached run of the phase dir ``sys.argv[2]`` that its abort can stop:
#: the production RunHandle, with its signal handlers, in a session of its own.
_STOPPABLE_RUN = r"""
import os, sys, time
sys.path.insert(0, sys.argv[1])
from utils import detach
handle = detach.RunHandle(sys.argv[2], run_id=sys.argv[3], mode="review-code",
                          plan_file=sys.argv[4], detached=False)
handle.start()
print("running", flush=True)
time.sleep(120)
"""


def _inode(path: Path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino


class TestOneWriterPerPhaseDir:
    def _take(self, run, phase_dir, **flags):
        return run.take(phase_dir, mode="review-code", plan_file=str(phase_dir / "p.md"),
                        args=_args(**flags), phase="code-review")

    def test_the_hold_lasts_the_run_and_is_released_at_its_exit(self, tmp_path):
        from utils import detach, process_control

        lock = str(tmp_path / ".lock")
        with detach.PhaseRun() as run:
            self._take(run, tmp_path)
            assert process_control.held_phase_lock(lock) is not None
            assert not process_control.lock_is_free(lock)
            holder = process_control.read_lock_holder(lock)
            assert holder["pid"] == os.getpid() and holder["mode"] == "review-code"
        assert process_control.held_phase_lock(lock) is None
        assert process_control.lock_is_free(lock)

    def test_a_hold_in_one_process_is_re_entrant(self, tmp_path):
        """A phase main under a composite step's or a front door's lock, or a
        nested run of the same dir: the inner take uses the held lock (the OS
        would refuse this process a second descriptor) and never releases it."""
        from utils import detach, process_control

        lock_file = str(tmp_path / ".lock")
        outer = process_control.PhaseLock(lock_file)
        outer.acquire({"holder": "improve step", "pid": os.getpid()})
        try:
            with detach.PhaseRun() as run:
                self._take(run, tmp_path)
                with detach.PhaseRun() as inner:
                    self._take(inner, tmp_path)
            assert outer.held
            assert process_control.held_phase_lock(lock_file) is outer
            assert process_control.read_lock_holder(lock_file)["holder"] == "improve step"
        finally:
            outer.release()
        assert process_control.lock_is_free(lock_file)

    def test_the_worker_hands_its_entry_hold_to_its_run_handle(self, tmp_path):
        """The detached worker takes the lock at its entry, so its pre-flight
        runs under it; its RunHandle takes that hold over (no second take) and
        releases it after the terminal record, not at the phase main's exit."""
        from utils import detach, process_control

        plan = tmp_path / "plan.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        lock_file = str(tmp_path / ".lock")
        handle = detach.RunHandle(str(tmp_path), run_id="run-1", mode="review-code",
                                  plan_file=str(plan), detached=False)
        with detach.PhaseRun() as run:
            self._take(run, tmp_path, internal_worker=True, internal_run_id="run-1")
            entry = process_control.held_phase_lock(lock_file)
            handle._acquire_lock()
            assert handle._lock is entry
            assert process_control.read_lock_holder(lock_file)["run_id"] == "run-1"
        assert handle._lock.held, "the worker's hold outlives the phase main"
        handle._lock.release()
        assert process_control.lock_is_free(lock_file)

    def test_a_stale_lock_left_by_a_dead_process_is_taken_over(self, tmp_path):
        """A holder that died released its lock with its descriptor; the label
        it left names a dead process and is overwritten (the existing rule:
        the lock is the authority, the payload a label)."""
        import subprocess

        from utils import detach, process_control

        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        _write(tmp_path / ".lock", {"run_id": "dead-run", "pid": gone.pid,
                                    "process_start_time": 1.0})
        assert process_control.holder_verdict(
            process_control.read_lock_holder(str(tmp_path / ".lock"))) == process_control.DEAD
        with detach.PhaseRun() as run:
            self._take(run, tmp_path)
            assert process_control.read_lock_holder(str(tmp_path / ".lock"))["pid"] == os.getpid()

    def test_a_live_holder_refuses_every_lane_at_once_and_names_its_pid(
            self, tmp_path, capsys):
        """The lock is not waited on: after the few retries that ride out a
        `status` probe, a live holder refuses a foreground run, a worker and a
        launcher at exit 3, before any of them reads or writes anything."""
        import time

        from utils import detach

        holder = _hold_in_another_process(tmp_path / ".lock")
        try:
            started = time.monotonic()
            with detach.PhaseRun() as run, pytest.raises(SystemExit) as stop:
                self._take(run, tmp_path)
            assert time.monotonic() - started < 5
            assert stop.value.code == 3
            out = capsys.readouterr().out
            assert f"another run of this phase is in progress (pid {holder.pid}, alive)" in out
            assert "review-code (foreground) owns this phase dir" in out
            with detach.PhaseRun() as run, pytest.raises(SystemExit) as stop:
                self._take(run, tmp_path, internal_worker=True, internal_run_id="r")
            assert stop.value.code == 3
            # A launcher is refused at its entry too: nothing of its
            # pre-flight runs (Codex round 3, point 3).
            with detach.PhaseRun() as run, pytest.raises(SystemExit) as stop:
                self._take(run, tmp_path, detach=True)
            assert stop.value.code == 3
            err = capsys.readouterr().err
            assert f"is held by review-code (foreground) (pid {holder.pid}, alive)" in err
            assert "nothing was launched" in err
        finally:
            holder.communicate("\n", timeout=10)
        assert sorted(path.name for path in tmp_path.iterdir()) == [".lock"]

    def test_a_lock_that_cannot_be_taken_refuses_every_lane(self, tmp_path, capsys):
        """Fail closed: a lock file that cannot be opened is never a licence to
        run unlocked."""
        from utils import detach

        (tmp_path / ".lock").mkdir()
        for flags in ({}, {"internal_worker": True}, {"detach": True}):
            with detach.PhaseRun() as run, pytest.raises(SystemExit) as stop:
                self._take(run, tmp_path, **flags)
            assert stop.value.code == 3
        assert "could not lock" in capsys.readouterr().out

    def test_no_child_inherits_the_lock(self, tmp_path):
        """The lock's descriptor is close-on-exec: a provider CLI, even one
        started without `close_fds`, never keeps the phase locked."""
        import subprocess

        from utils import detach, process_control

        lock_file = tmp_path / ".lock"
        with detach.PhaseRun() as run:
            self._take(run, tmp_path)
            held = process_control.held_phase_lock(str(lock_file))
            assert os.get_inheritable(held._fd) is False
            listing = subprocess.run(
                [sys.executable, "-c",
                 "import os\n"
                 "for fd in range(3, 256):\n"
                 "    try: print(os.readlink(f'/proc/self/fd/{fd}'))\n"
                 "    except OSError: pass\n"],
                close_fds=False, capture_output=True, text=True, check=True,
            ).stdout if os.path.isdir("/proc/self/fd") else ""
            assert str(lock_file) not in listing
        assert process_control.lock_is_free(str(lock_file))

    def test_a_foreground_run_beside_a_held_lock_writes_nothing(self, tmp_path):
        """The orchestrator end to end: a plain review-code while another
        process holds the dir exits 3 and leaves every file as it was."""
        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        state = StateManager(plan)
        state.state["phases_completed"].pop("code-review")
        state.save()
        holder = _hold_in_another_process(phase_dir / ".lock")
        try:
            before = _snapshot(plan)
            result = _run(runner, "code_review", plan)
            after = _snapshot(plan)
        finally:
            holder.communicate("\n", timeout=10)
        _refused_at_the_lock(result, holder.pid)
        assert "Auto-healed" not in result.stdout
        assert after == before


class TestABusyLauncherStopsAtItsEntry:
    """Codex round 3, point 3: a `--detach` launcher whose dir a live run
    holds used to run its pre-flight anyway, overwriting the live run's
    `backup.md` / `tasks-backup.md` before its launch refused. The decision is
    now the entry gate's, before any pre-flight write."""

    @pytest.mark.parametrize("mode, phase, backup", [
        ("review-plan", "review-plan", "backup.md"),
        ("review-tasks", "review-tasks", "tasks-backup.md"),
        ("review-code", "code-review", None),
    ])
    def test_a_busy_launcher_changes_nothing(self, tmp_path, mode, phase, backup):
        runner = SkillRunner(tmp_path / "harness")
        runner.set_config({"response": [SUGGESTION]})
        plan = _plan(tmp_path)
        if mode == "review-code":
            _implemented(plan)
        phase_dir = _phase(plan, phase)
        phase_dir.mkdir(parents=True, exist_ok=True)
        if backup:
            (phase_dir / backup).write_text("THE HELD RUN'S SNAPSHOT", encoding="utf-8")
        holder = _hold_in_another_process(phase_dir / ".lock")
        try:
            before = _snapshot(plan)
            result = runner._run_command(
                [sys.executable, str(SKILL_ROOT / "multi_llm.py"), mode, str(plan),
                 "--models", MODEL, "--detach"], plan, 60, False)
            after = _snapshot(plan)
        finally:
            holder.communicate("\n", timeout=10)
        assert after == before
        assert result.exit_code == 3, result.stdout + result.stderr
        assert f"is held by review-code (foreground) (pid {holder.pid}" in result.stderr

    def test_if_not_running_re_attaches_from_the_checkpoint_alone(self, tmp_path, capsys):
        """`--if-not-running` onto the live detached run that holds the dir:
        decided from its lock label and checkpoint, writing nothing."""
        from utils import detach

        holder = _hold_in_another_process(tmp_path / ".lock", run_id="run-held-0001")
        try:
            before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
            with detach.PhaseRun() as run, pytest.raises(SystemExit) as stop:
                run.take(tmp_path, mode="review-code", plan_file=str(tmp_path / "p.md"),
                         args=_args(detach=True, if_not_running=True), phase="code-review")
            after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        finally:
            holder.communicate("\n", timeout=10)
        assert stop.value.code == 0
        out = capsys.readouterr().out
        assert "already in flight" in out and "run-held-0001" in out
        assert after == before


class TestForceNeverBreaksALiveLock:
    """Codex round 3, point 1: `--detach --force` unlinked a live holder's
    lock, so a paused reaggregation completed a phase another round had
    taken over. A live lock is never broken: `--force` stops the detached
    run that holds it through the abort path, then waits for the same lock."""

    def _launch_take(self, phase_dir: Path, **flags):
        from utils import detach

        run = detach.PhaseRun()
        run.take(phase_dir, mode="review-code", plan_file=str(phase_dir / "p.md"),
                 args=_args(detach=True, **flags), phase="code-review")
        return run

    def test_force_stops_the_detached_holder_then_takes_its_lock(self, tmp_path, capsys):
        import subprocess
        import threading

        from utils import process_control, status_file

        plan = tmp_path / "p.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        holder = subprocess.Popen(
            [sys.executable, "-c", _STOPPABLE_RUN, str(SKILL_ROOT), str(tmp_path),
             "run-stop-0001", str(plan)],
            stdout=subprocess.PIPE, text=True, start_new_session=True,
        )
        try:
            while holder.stdout.readline().strip() != "running":
                assert holder.poll() is None
            # Reaped the moment it exits, as init reaps a real detached worker:
            # the abort confirms the stop only once the pid is gone, and an
            # unreaped child of this process would read as `survived`.
            threading.Thread(target=holder.wait, daemon=True).start()
            lock_file = tmp_path / ".lock"
            inode = _inode(lock_file)
            run = self._launch_take(tmp_path, force=True)
            try:
                assert holder.wait(timeout=30) == 3, "stopped through its abort path"
                assert _inode(lock_file) == inode, "the lock file is never unlinked"
                label = process_control.read_lock_holder(str(lock_file))
                assert label["pid"] == os.getpid()
                assert "--detach launcher" in label["holder"]
                assert status_file.read_run(str(tmp_path))["state"] == "aborted"
            finally:
                run.__exit__(None, None, None)
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
        err = capsys.readouterr().err
        assert "stopping run run-stop-0001" in err and "has stopped" in err

    def test_force_refuses_a_holder_it_cannot_stop(self, tmp_path, capsys):
        """A foreground run (or a composite's step, or another launcher) is
        not a run `--force` can stop: refused at once, its lock intact."""
        from utils import process_control

        holder = _hold_in_another_process(tmp_path / ".lock")
        try:
            inode = _inode(tmp_path / ".lock")
            with pytest.raises(SystemExit) as stop:
                self._launch_take(tmp_path, force=True)
            assert stop.value.code == 3
            assert holder.poll() is None, "never signalled"
            assert not process_control.lock_is_free(str(tmp_path / ".lock"))
            assert _inode(tmp_path / ".lock") == inode
        finally:
            holder.communicate("\n", timeout=10)
        err = capsys.readouterr().err
        assert "--force stops only a detached run" in err
        assert f"pid {holder.pid}" in err

    def test_a_holder_that_outlives_the_wait_is_refused_and_keeps_its_lock(
            self, tmp_path, capsys, monkeypatch):
        from utils import detach, process_control

        asked = []
        monkeypatch.setattr(detach, "abort_run",
                            lambda phase_dir, **kw: asked.append(phase_dir) or {"result": "aborted"})
        monkeypatch.setattr(detach, "FORCE_STOP_WAIT_S", 0.3)
        holder = _hold_in_another_process(tmp_path / ".lock", run_id="run-stuck-0001")
        try:
            with pytest.raises(SystemExit) as stop:
                self._launch_take(tmp_path, force=True)
            assert stop.value.code == 3
            assert asked == [str(tmp_path)]
            assert holder.poll() is None
            assert process_control.read_lock_holder(str(tmp_path / ".lock"))["run_id"] == (
                "run-stuck-0001")
            assert not process_control.lock_is_free(str(tmp_path / ".lock"))
        finally:
            holder.communicate("\n", timeout=10)
        err = capsys.readouterr().err
        assert "still held after 0s by detached run run-stuck-0001" in err
        assert "never broken" in err

    @pytest.mark.parametrize("stopped", [
        {"result": "incomplete", "verdict": "unresolved"},
        {"result": "unresolved"},
        {"result": "survived"},
        {"result": "an-outcome-added-later"},
        {},
    ], ids=["incomplete", "unresolved", "survived", "unknown", "no-result"])
    def test_only_a_confirmed_stop_lets_force_take_the_dir_over(
            self, tmp_path, capsys, monkeypatch, stopped):
        """Codex round 4, point 2: an abort that could not confirm every
        process of the run stopped still let `--force` take the lock once the
        worker had died, and a provider child that outlived it then replaced
        the new round's result. `--force` takes over only after `aborted` or
        `not_running`; any other outcome refuses, one it does not know too."""
        from utils import detach, process_control

        lock_file = tmp_path / ".lock"
        holder = _hold_in_another_process(lock_file, run_id="run-left-0001")
        records = (tmp_path / ".status.json").read_bytes()

        def abort(phase_dir, **_kw):
            holder.communicate("\n", timeout=10)  # the worker is gone, and its lock
            return dict(stopped, phase_dir=phase_dir)

        monkeypatch.setattr(detach, "abort_run", abort)
        try:
            with pytest.raises(SystemExit) as stop:
                self._launch_take(tmp_path, force=True)
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
        assert stop.value.code == 3
        assert process_control.held_phase_lock(str(lock_file)) is None, "never taken"
        assert (tmp_path / ".status.json").read_bytes() == records, "records kept"
        err = capsys.readouterr().err
        assert "not every process of that run is confirmed stopped" in err
        assert "Nothing was launched" in err
        assert "--abort --run-id run-left-0001" in err
        assert "runs --sweep" in err

    @pytest.mark.parametrize("result", ["aborted", "not_running"])
    def test_a_confirmed_stop_lets_force_take_the_dir_over(
            self, tmp_path, monkeypatch, result):
        from utils import detach, process_control

        lock_file = tmp_path / ".lock"
        holder = _hold_in_another_process(lock_file, run_id="run-gone-0001")

        def abort(phase_dir, **_kw):
            holder.communicate("\n", timeout=10)
            return {"result": result, "phase_dir": phase_dir}

        monkeypatch.setattr(detach, "abort_run", abort)
        try:
            run = self._launch_take(tmp_path, force=True)
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
        try:
            assert process_control.held_phase_lock(str(lock_file)) is not None
            assert process_control.read_lock_holder(str(lock_file))["pid"] == os.getpid()
        finally:
            run.__exit__(None, None, None)

    def test_break_lock_never_frees_a_live_holder(self, tmp_path):
        from utils import process_control

        holder = _hold_in_another_process(tmp_path / ".lock", run_id="run-live-0001")
        try:
            inode = _inode(tmp_path / ".lock")
            assert process_control.break_lock(str(tmp_path / ".lock")) is False
            assert _inode(tmp_path / ".lock") == inode
            assert not process_control.lock_is_free(str(tmp_path / ".lock"))
        finally:
            holder.communicate("\n", timeout=10)


#: A stand-in for `uv run … --internal-worker`: the worker side of a launch,
#: driving the production PhaseRun and RunHandle. Before it adopts the lock
#: and after, it asks a competing process to take it.
_WORKER_SHIM = r"""#!{python}
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, {skill!r})
from utils import detach

PHASE, PLAN, REPORT, DONE = {phase!r}, {plan!r}, {report!r}, {done!r}
run_id = sys.argv[sys.argv.index("--internal-run-id") + 1]

def competitor():
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]);"
         "from utils import process_control as pc;"
         "lock = pc.PhaseLock(sys.argv[2])\n"
         "try:\n lock.acquire(); print('acquired')\n"
         "except pc.LockUnavailable:\n print('refused')",
         {skill!r}, os.path.join(PHASE, ".lock")],
        capture_output=True, text=True).stdout.strip()

seen = {{"before_adoption": competitor(), "fd_env": os.environ.get(detach.LOCK_FD_ENV)}}
detach.prepare_worker_output()
with detach.PhaseRun() as run:
    run.take(PHASE, mode="review-plan", plan_file=PLAN, phase="review-plan",
             args=argparse.Namespace(internal_worker=True, internal_run_id=run_id,
                                     host=None, detach=False))
    handle = detach.RunHandle(PHASE, run_id=run_id, mode="review-plan", plan_file=PLAN)
    handle.start()
seen["after_adoption"] = competitor()
seen["env_left"] = os.environ.get(detach.LOCK_FD_ENV)
with open(REPORT, "w") as out:
    json.dump(seen, out)
deadline = time.time() + 60
while not os.path.exists(DONE) and time.time() < deadline:
    time.sleep(0.05)
handle.finish(0)
"""


def _competitor(lock_file: Path) -> str:
    import subprocess

    return subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]);"
         "from utils import process_control as pc;"
         "lock = pc.PhaseLock(sys.argv[2])\n"
         "try:\n lock.acquire(); print('acquired')\n"
         "except pc.LockUnavailable:\n print('refused')",
         str(SKILL_ROOT), str(lock_file)],
        capture_output=True, text=True, check=True).stdout.strip()


class TestTheLockCrossesTheDetachBoundaryHeld:
    """Codex round 3, point 4: the launcher dropped its hold at the boundary
    and its worker took the lock again, so another process could take it in
    between and the worker was refused. On POSIX the worker now inherits the
    launcher's open file description and adopts it; the launcher only ever
    closes its copy."""

    @POSIX_ONLY
    def test_a_competitor_never_gets_the_lock_from_entry_to_worker(
            self, tmp_path, monkeypatch, capsys):
        import time

        from utils import detach, status_file

        plan = tmp_path / "p.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        phase_dir = tmp_path / "p" / "review-plan"
        phase_dir.mkdir(parents=True)
        lock_file = phase_dir / ".lock"
        report, done = tmp_path / "worker.json", tmp_path / "done"
        shim = tmp_path / "uv-shim"
        shim.write_text(_WORKER_SHIM.format(
            python=sys.executable, skill=str(SKILL_ROOT), phase=str(phase_dir),
            plan=str(plan), report=str(report), done=str(done)), encoding="utf-8")
        shim.chmod(0o755)
        monkeypatch.setenv(detach.UV_ENV, str(shim))
        monkeypatch.setenv(detach.HEARTBEAT_INTERVAL_ENV, "0.2")
        args = _args(detach=True, max_detached=None)
        seen = {}
        with detach.PhaseRun() as run:
            run.take(phase_dir, mode="review-plan", plan_file=str(plan), args=args,
                     phase="review-plan")
            seen["launcher_preflight"] = _competitor(lock_file)
            code = detach.launch(mode="review-plan", argv=["--plan-file", str(plan)],
                                 phase_dir=str(phase_dir), plan_file=str(plan),
                                 phase="review-plan", args=args)
            seen["after_launch"] = _competitor(lock_file)
        seen["launcher_gone"] = _competitor(lock_file)
        capsys.readouterr()
        worker = json.loads(report.read_text(encoding="utf-8"))
        done.write_text("go", encoding="utf-8")
        deadline = time.monotonic() + 30
        while status_file.read_run(str(phase_dir)).get("state") != "done":
            assert time.monotonic() < deadline, status_file.read_run(str(phase_dir))
            time.sleep(0.05)
        while _competitor(lock_file) != "acquired":
            assert time.monotonic() < deadline, "the worker never let go"
            time.sleep(0.05)

        assert code == 0
        assert seen == {"launcher_preflight": "refused", "after_launch": "refused",
                        "launcher_gone": "refused"}
        assert worker["fd_env"] is not None
        assert worker["before_adoption"] == "refused"
        assert worker["after_adoption"] == "refused"
        assert worker["env_left"] is None, "no provider child sees the descriptor"

    def test_a_descriptor_that_is_not_the_lock_is_refused(self, tmp_path, monkeypatch, capsys):
        from utils import detach, process_control

        other = tmp_path / "other.txt"
        other.write_text("x", encoding="utf-8")
        fd = os.open(str(other), os.O_RDONLY)
        try:
            (tmp_path / ".lock").write_text("", encoding="utf-8")
            with pytest.raises(process_control.LockHandoffFailed):
                process_control.PhaseLock.adopt(str(tmp_path / ".lock"), fd)
            monkeypatch.setenv(detach.LOCK_FD_ENV, str(fd))
            with detach.PhaseRun() as run, pytest.raises(SystemExit) as stop:
                run.take(tmp_path, mode="review-code", plan_file=str(tmp_path / "p.md"),
                         args=_args(internal_worker=True, internal_run_id="r"),
                         phase="code-review")
            assert stop.value.code == 3
            assert "is not" in capsys.readouterr().out
            os.fstat(fd)  # left open: a number naming another file is not ours to close
        finally:
            os.close(fd)

    def test_a_worker_refused_at_the_lock_is_reported_not_launched(
            self, tmp_path, monkeypatch, capsys):
        """Windows retakes the lock across the boundary; a worker that finds
        it taken leaves a refusal record and the launch reports it, rather
        than a run. Both halves, on any platform."""
        import argparse

        from utils import detach, process_control

        holder = _hold_in_another_process(tmp_path / ".lock")
        monkeypatch.setenv(detach.WORKER_ENV, "1")
        try:
            with detach.PhaseRun() as run, pytest.raises(SystemExit) as stop:
                run.take(tmp_path, mode="review-code", plan_file=str(tmp_path / "p.md"),
                         args=_args(internal_worker=True, internal_run_id="run-refused-1"),
                         phase="code-review")
        finally:
            holder.communicate("\n", timeout=10)
        assert stop.value.code == 3
        record = tmp_path / ".worker-refused-run-refused-1"
        assert json.loads(record.read_text(encoding="utf-8"))["message"].startswith(
            "ERROR: another run of this phase is in progress")
        monkeypatch.delenv(detach.WORKER_ENV)

        refusal = record.read_bytes()
        record.unlink()
        plan = tmp_path / "p.md"
        plan.write_text("# Plan\n", encoding="utf-8")
        monkeypatch.setenv(detach.UV_ENV, sys.executable)

        def spawn(command, *, cwd, env, log_path):
            run_id = command[command.index("--internal-run-id") + 1]
            (tmp_path / f".worker-refused-{run_id}").write_bytes(refusal)
            return os.getpid()

        monkeypatch.setattr(process_control, "spawn_detached", spawn)
        with pytest.raises(detach.DetachError) as refused:
            detach.launch(mode="review-code", argv=["--plan-file", str(plan)],
                          phase_dir=str(tmp_path), plan_file=str(plan), phase="code-review",
                          args=argparse.Namespace(force=False, if_not_running=False,
                                                  max_detached=None))
        assert refused.value.exit_code == 3
        assert "stopped before it took" in str(refused.value)
        assert "another run of this phase is in progress" in str(refused.value)
        assert not list(tmp_path.glob(".worker-refused-*")), "read once, then removed"
        assert process_control.lock_is_free(str(tmp_path / ".lock"))


#: A review-code `--reaggregate --exec-subagents` whose executor is held right
#: before it persists validation batch 0, until `sys.argv[3]` exists.
_HELD_PERSIST = r"""
import os, sys, time
from pathlib import Path
sys.path[:0] = [sys.argv[1]]
from utils import task_result
plan, gate, ready = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
persist = task_result.persist_result
def held(task, phase_dir, data):
    if str(task.get("output_name", "")).endswith("validation_batch_0.json"):
        ready.write_text("ready")
        while not gate.exists():
            time.sleep(0.02)
        result = persist(task, phase_dir, data)
        (Path(phase_dir) / "late-write").write_text("written")
        return result
    return persist(task, phase_dir, data)
task_result.persist_result = held
import multi_llm
sys.exit(multi_llm.main(["review-code", plan, "--reaggregate", "--yes", "--exec-subagents"]))
"""


class TestACtrlCHoldsTheLockUntilItsDispatchEnds:
    """Codex round 3, point 2: a Ctrl-C cancelled the phase main, which
    released the lock while its dispatch thread was still running; another
    round then completed, and the old dispatch overwrote its batch."""

    @POSIX_ONLY
    def test_the_lock_outlives_a_cancelled_dispatch(self, tmp_path):
        import signal
        import subprocess
        import time

        from utils import process_control

        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        for path in phase_dir.glob("validation_batch_*.json"):
            path.unlink()
        runner.extra_env["MOCK_LLM_ECHO"] = "validation"
        gate, ready = tmp_path / "gate", tmp_path / "ready"
        child = subprocess.Popen(
            [sys.executable, "-c", _HELD_PERSIST, str(SKILL_ROOT), str(plan), str(gate),
             str(ready)],
            env=runner._build_env(), cwd=str(SKILL_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        lock = str(phase_dir / ".lock")
        try:
            _wait_for(ready)
            child.send_signal(signal.SIGINT)
            freed_early = False
            until = time.monotonic() + 2.0
            while time.monotonic() < until:
                freed_early = freed_early or process_control.lock_is_free(lock)
                time.sleep(0.05)
            assert child.poll() is None, "the run waits for its dispatch"
            gate.write_text("go", encoding="utf-8")
            out, _ = child.communicate(timeout=60)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        assert not freed_early, out
        assert (phase_dir / "late-write").exists(), "the dispatch ran to its end"
        assert process_control.lock_is_free(lock), out

    def test_a_dispatch_still_running_past_the_wait_holds_the_lock_until_exit(
            self, tmp_path, monkeypatch):
        import asyncio
        import threading
        import time

        from utils import detach, process_control

        monkeypatch.setattr(detach, "PHASE_WORK_JOIN_S", 0.2)
        monkeypatch.setattr(detach, "PHASE_WORK_RESWEEP_S", 0.05)
        release = threading.Event()
        lock_file = str(tmp_path / ".lock")
        hold = process_control.PhaseLock.hold_until_exit

        def held_then_released(lock):
            # The work ends only once its run gave up waiting: the loop's
            # executor shutdown in `asyncio.run` joins it after that.
            hold(lock)
            release.set()

        monkeypatch.setattr(process_control.PhaseLock, "hold_until_exit", held_then_released)

        async def phase_main():
            with detach.PhaseRun() as run:
                run.take(tmp_path, mode="review-code", plan_file=str(tmp_path / "p.md"),
                         args=_args(), phase="code-review")
                await detach.run_phase_work(release.wait, 30)

        async def cancelled():
            task = asyncio.ensure_future(phase_main())
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancelled())
        held = process_control.held_phase_lock(lock_file)
        try:
            assert release.is_set(), "the wait gave up while the work was running"
            assert held is not None and not process_control.lock_is_free(lock_file)
            time.sleep(0.2)
            held.release()
            assert not process_control.lock_is_free(lock_file), "held until exit"
        finally:
            release.set()
            held._until_exit = False
            held.release()
        assert process_control.lock_is_free(lock_file)

    def test_a_dispatch_that_ends_in_the_wait_lets_the_lock_go(self, tmp_path):
        import asyncio
        import threading

        from utils import detach, process_control

        release = threading.Event()
        lock_file = str(tmp_path / ".lock")

        async def phase_main():
            with detach.PhaseRun() as run:
                run.take(tmp_path, mode="review-code", plan_file=str(tmp_path / "p.md"),
                         args=_args(), phase="code-review")
                await detach.run_phase_work(release.wait, 30)

        async def cancelled():
            task = asyncio.ensure_future(phase_main())
            await asyncio.sleep(0.1)
            threading.Timer(0.3, release.set).start()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert release.is_set(), "the exit waited for the dispatch"

        asyncio.run(cancelled())
        assert process_control.lock_is_free(lock_file)

    def test_a_late_result_of_a_replaced_round_is_dropped(self, tmp_path, capsys):
        """The executor persists a batch result only while the round that
        dispatched it is the round on record."""
        from utils.subagent_executor import persist_result

        claim = review_round.start_round(tmp_path)
        task = {"task_id": "v0", "kind": "validation",
                "output_name": "validation_batch_0.json"}
        with review_round.results_bound_to(claim):
            persist_result(task, str(tmp_path), {"groups": ["first"]})
            assert json.loads((tmp_path / "validation_batch_0.json").read_text()) == {
                "groups": ["first"]}
            other = review_round.start_round(tmp_path)
            persist_result(task, str(tmp_path), {"groups": ["late"]})
        assert json.loads((tmp_path / "validation_batch_0.json").read_text()) == {
            "groups": ["first"]}
        err = capsys.readouterr().err
        assert "validation_batch_0.json dropped, not written" in err
        assert other.round_id in err
        # Unbound, a write is what it always was.
        persist_result(task, str(tmp_path), {"groups": ["unbound"]})
        assert json.loads((tmp_path / "validation_batch_0.json").read_text()) == {
            "groups": ["unbound"]}

    def test_the_binding_lasts_exactly_as_long_as_the_dispatch(self, tmp_path):
        import asyncio

        from utils import detach

        claim = review_round.start_round(tmp_path)
        bound_inside = []

        def dispatch(phase_dir):
            bound_inside.append(list(review_round._BOUND.get(review_round._bound_key(phase_dir))))
            return "done"

        result = asyncio.run(detach.run_phase_work(
            review_round.bound(claim, dispatch), str(tmp_path)))
        assert result == "done"
        assert bound_inside == [[claim]]
        assert review_round._bound_key(tmp_path) not in review_round._BOUND


#: A review-code `--force --rerun-all` whose fan-out thread is held, until
#: `sys.argv[3]` exists, right before it writes the provider log (`log`) or
#: sets the previous round's unchanged result aside (`rename`).
_HELD_FANOUT = r"""
import json, sys, time
from pathlib import Path
sys.path[:0] = [sys.argv[1]]
from utils import llm_client, result_files
plan, gate, ready, where = sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
if where == "rename":
    answer = json.loads(sys.argv[6])
    llm_client.invoke_with_provider = lambda *a, **kw: {
        "success": True, "data": answer, "details": {}}
    module, name = result_files, "set_aside_stale_result"
else:
    module, name = llm_client, "_save_log"
real = getattr(module, name)
def held(*args, **kwargs):
    ready.write_text("ready")
    while not gate.exists():
        time.sleep(0.02)
    result = real(*args, **kwargs)
    (gate.parent / "late-write").write_text("written")
    return result
setattr(module, name, held)
import multi_llm
sys.exit(multi_llm.main(["review-code", plan, "--models", sys.argv[7], "--yes",
                         "--force", "--rerun-all"]))
"""


class TestACtrlCHoldsTheLockUntilTheFanOutEnds:
    """Codex round 4, point 1: the fan-out's own threads (a provider call,
    with the log it writes and the previous round's result it sets aside)
    were not waited for, so a Ctrl-C released the lock while one was still
    running; another round then ran, and the old thread overwrote its log
    or moved its result aside. Every thread a review phase starts goes
    through `review_orchestrator_base.phase_thread`."""

    @POSIX_ONLY
    @pytest.mark.parametrize("where", ["log", "rename"])
    def test_the_lock_outlives_a_cancelled_fan_out(self, tmp_path, where):
        import signal
        import subprocess
        import time

        from utils import process_control

        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        gate, ready = tmp_path / "gate", tmp_path / "ready"
        child = subprocess.Popen(
            [sys.executable, "-c", _HELD_FANOUT, str(SKILL_ROOT), str(plan), str(gate),
             str(ready), where, json.dumps(CODE_ISSUES), MODEL],
            env=runner._build_env(), cwd=str(SKILL_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        lock = str(phase_dir / ".lock")
        try:
            _wait_for(ready)
            child.send_signal(signal.SIGINT)
            freed_early = False
            until = time.monotonic() + 2.0
            while time.monotonic() < until:
                freed_early = freed_early or process_control.lock_is_free(lock)
                time.sleep(0.05)
            assert child.poll() is None, "the run waits for its fan-out thread"
            gate.write_text("go", encoding="utf-8")
            out, _ = child.communicate(timeout=60)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        assert not freed_early, out
        assert (tmp_path / "late-write").exists(), "the thread ran to its end"
        assert process_control.lock_is_free(lock), out

    def test_no_review_phase_module_starts_a_thread_of_its_own(self):
        """The rule the test above relies on, for every site: the review
        base, review-code, the shared review lane (whose slot core dispatches
        through the lane's `phase_thread`) and the slot guard start threads only
        through `phase_thread` (and `dispatch_in_round` over it)."""
        import ast

        offenders = []
        for rel in ("utils/review_orchestrator_base.py", "code_review_orchestrator.py",
                    "utils/review_lane.py", "utils/slot_guard.py"):
            tree = ast.parse((SKILL_ROOT / rel).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = func.attr if isinstance(func, ast.Attribute) else getattr(
                        func, "id", "")
                    if name in {"to_thread", "run_in_executor", "Thread",
                                "ThreadPoolExecutor"}:
                        offenders.append(f"{rel}:{node.lineno} {name}")
        assert offenders == []

    def test_a_call_given_up_before_a_thread_started_it_is_not_waited_for(self):
        import asyncio
        import concurrent.futures
        import threading

        from utils import detach

        gate = threading.Event()
        ran = []

        async def scenario():
            loop = asyncio.get_running_loop()
            loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
            first = asyncio.ensure_future(detach.run_phase_work(gate.wait, 10))
            await asyncio.sleep(0.05)
            queued = asyncio.ensure_future(detach.run_phase_work(ran.append, "ran"))
            await asyncio.sleep(0.05)
            assert len(detach._pending_phase_work()) == 2
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert len(detach._pending_phase_work()) == 1, "only the running call"
            gate.set()
            assert await first is True

        try:
            asyncio.run(scenario())
        finally:
            gate.set()
        assert ran == []
        assert detach._pending_phase_work() == []

    def test_an_interrupted_wait_stops_a_cli_started_after_its_first_sweep(
            self, tmp_path, monkeypatch):
        """A thread that starts its provider CLI after the interrupt's sweep
        is stopped by the next one, so the run lets its lock go when the
        thread ends instead of holding it until the process exits."""
        import asyncio
        import threading

        from utils import detach, process_control

        monkeypatch.setattr(detach, "PHASE_WORK_JOIN_S", 5.0)
        monkeypatch.setattr(detach, "PHASE_WORK_RESWEEP_S", 0.05)
        cli_ended = threading.Event()
        sweeps = []

        def sweep(tag):
            sweeps.append(tag)
            if len(sweeps) >= 2:
                cli_ended.set()  # the CLI the work started late is stopped now

        monkeypatch.setattr(process_control, "sweep_live_clis", sweep)
        lock_file = str(tmp_path / ".lock")

        async def phase_main():
            with detach.PhaseRun() as run:
                run.take(tmp_path, mode="review-code", plan_file=str(tmp_path / "p.md"),
                         args=_args(), phase="code-review")
                await detach.run_phase_work(cli_ended.wait, 30)

        async def cancelled():
            task = asyncio.ensure_future(phase_main())
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        try:
            asyncio.run(cancelled())
        finally:
            cli_ended.set()
        assert len(sweeps) >= 2 and set(sweeps) == {"INTERRUPT_SWEEP"}
        assert process_control.lock_is_free(lock_file), "released once the work ended"

    @pytest.mark.parametrize("first", ["the-awaiter", "the-thread"])
    def test_a_call_is_given_up_or_waited_for_never_lost_between(
            self, tmp_path, monkeypatch, first):
        """Codex round 5, point 1 (u81-r5/start_cancel_race.py): a cancelled
        awaiter read "not started" under the mutex, let it go, then unlisted
        the call; a pool thread that started it in that gap ran a writer
        nobody waited for, and the lock went free under it. The scheduling
        lock below lets the thread in at the awaiter's first release after
        the cancel (where that gap was), or lets the thread claim the call
        before the cancel: either the writer never runs, or the lock is held
        until it ends."""
        import asyncio
        import concurrent.futures
        import threading

        from utils import detach, process_control

        monkeypatch.setattr(process_control, "sweep_live_clis", lambda tag: None)
        lock_file = str(tmp_path / ".lock")
        loop_thread = threading.current_thread()  # asyncio.run runs its loop here
        thread_at_lock, let_thread_in, thread_passed = (threading.Event() for _ in range(3))
        armed, writer_started, finish_writer = (threading.Event() for _ in range(3))
        wrote: list = []
        seen: Dict[str, Any] = {}

        class SchedulingLock:
            """`_PHASE_WORK_LOCK`, with the pool thread held at its door until
            the loop thread lets it in."""

            def __init__(self) -> None:
                self._lock = threading.Lock()
                self._fired = False

            def __enter__(self):
                if threading.current_thread() is not loop_thread:
                    thread_at_lock.set()
                    let_thread_in.wait(10)
                self._lock.acquire()
                return self

            def __exit__(self, *exc):
                self._lock.release()
                if threading.current_thread() is not loop_thread:
                    thread_passed.set()
                elif armed.is_set() and not self._fired:
                    self._fired = True
                    let_thread_in.set()
                    thread_passed.wait(10)
                return False

        monkeypatch.setattr(detach, "_PHASE_WORK_LOCK", SchedulingLock())

        def writer():
            writer_started.set()
            finish_writer.wait(10)
            wrote.append("the old round's file")

        def watch_the_lock():
            writer_started.wait(10)
            seen["free_while_writing"] = process_control.lock_is_free(lock_file)
            finish_writer.set()

        async def scenario():
            loop = asyncio.get_running_loop()
            loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
            await asyncio.to_thread(lambda: None)  # the pool's thread is already running

            async def phase_main():
                with detach.PhaseRun() as run:
                    run.take(tmp_path, mode="review-code", plan_file=str(tmp_path / "p.md"),
                             args=_args(), phase="code-review")
                    await detach.run_phase_work(writer)

            task = asyncio.ensure_future(phase_main())
            while not thread_at_lock.is_set():
                await asyncio.sleep(0.001)
            if first == "the-awaiter":
                armed.set()
            else:
                let_thread_in.set()
                while not writer_started.is_set():
                    await asyncio.sleep(0.001)
                threading.Thread(target=watch_the_lock, daemon=True).start()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            seen["started"] = writer_started.is_set()
            seen["pending"] = len(detach._pending_phase_work())
            seen["free_after"] = process_control.lock_is_free(lock_file)
            finish_writer.set()

        try:
            asyncio.run(scenario())
        finally:
            let_thread_in.set()
            finish_writer.set()
        assert seen["pending"] == 0 and seen["free_after"]
        if first == "the-awaiter":
            assert not seen["started"], "a call given up is never run"
            assert wrote == []
        else:
            assert seen["started"]
            assert seen["free_while_writing"] is False, "held until the writer ended"
            assert wrote == ["the old round's file"]


class TestTheSealBindsTheBytesTheReportCameFrom:
    def test_verdicts_replaced_after_the_merge_do_not_pass_for_the_report_s(self, tmp_path):
        """Codex point 5: reaggregation A merged valid verdicts and built its
        report; B then published other verdicts of the same round; A's seal
        hashed B's file and a plain review-code auto-healed on it. The digest
        is now the one of the bytes A wrote."""
        _write(tmp_path / ".status.json", {"round_id": R1})
        claim = review_round.Claim(str(tmp_path), R1)
        validation = tmp_path / "validation.json"
        mine = json.dumps({"groups": [{"group_index": 0, "status": "valid"}],
                           "metadata": {"round_id": R1}}, indent=2)
        verdicts = claim.publish_verdicts(validation, mine)
        assert claim.verdicts == verdicts == (R1, review_round.digest(validation.read_bytes()))
        _write(validation, {"groups": [{"group_index": 0, "status": "invalid"}],
                            "metadata": {"round_id": R1}})

        text = f"{report_markdown.ESCAPED_TEXT_MARKER}\n# Report\n\nOne valid finding.\n"
        review_round.publish_report(tmp_path / "report.md", text, claim=claim,
                                    verdicts=claim.verdicts)

        seal = review_round.report_seal(tmp_path)
        assert (seal.round_id, seal.whole, seal.binds_validation) == (R1, True, False)

    def test_verdicts_read_for_the_report_are_the_bytes_read(self, tmp_path):
        claim = review_round.Claim(str(tmp_path), R1)
        validation = tmp_path / "validation.json"
        _write(validation, {"groups": [], "metadata": {"round_id": R1}})
        first = validation.read_bytes()
        assert claim.read_verdicts(validation) == first
        _write(validation, {"groups": [{"status": "invalid"}], "metadata": {"round_id": R1}})
        assert claim.verdicts.digest == review_round.digest(first)

    def test_the_bytes_written_are_the_bytes_hashed(self, tmp_path):
        """Text is written as text mode writes it, so the digest recorded is
        that of the file on disk on every platform."""
        target = tmp_path / "validation.json"
        written = review_round.replace_file(target, '{\n  "a": 1\n}')
        assert written == target.read_bytes()


class TestRecoveryInputsAreReplacedWhole:
    def test_grouped_report_html_and_report_data_are_renamed_into_place(self, tmp_path):
        """A guard or a reaggregate reads them back, so a writer stopped
        midway must leave the old file or the new one (rename, never an
        in-place rewrite)."""
        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        names = ("grouped.json", "report.html", "report_data.json")
        # Held open on POSIX so their inode numbers cannot be reused by the
        # files that replace them (a freed number can come back on ext4 under
        # load); Windows refuses a rename over an open file, so it only reads.
        held = {} if os.name == "nt" else {
            name: open(phase_dir / name, "rb") for name in names}
        try:
            inodes = {name: (phase_dir / name).stat().st_ino for name in names}

            _ok(_run(runner, "code_review", plan, "--force"))
            runner.simulate_subagents(phase_dir / "validation_tasks.json")
            _ok(runner.run_orchestrator("code_review", plan, "--reaggregate", "--yes",
                                        timeout=120))

            for name in names:
                assert (phase_dir / name).stat().st_ino != inodes[name], name
            assert not list(phase_dir.glob(".*.tmp"))
        finally:
            for handle in held.values():
                handle.close()

    def test_a_failed_html_write_keeps_the_old_page(self, tmp_path, monkeypatch):
        from utils import html_report_generator as html

        page = html.write_html_report("<p>old</p>", tmp_path)
        def refuse(*_args):
            raise OSError("injected replace failure")

        monkeypatch.setattr(os, "replace", refuse)
        with pytest.raises(OSError):
            html.write_html_report("<p>new</p>", tmp_path)
        assert page.read_text(encoding="utf-8") == "<p>old</p>"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["report.html"]


@pytest.mark.parametrize("phase", PHASE_NAMES)
class TestTheFirstStampEndsALegacyValidationRound:
    def test_legacy_batches_are_no_longer_resumed_as_the_new_round(self, tmp_path, phase):
        """Codex's fixture: id-less task file and batch 1 of 2 beside a
        ``validation_pending`` checkpoint (an upgraded dir), then a fan-out
        stamps its first id and stops. The stamp used to keep the old word,
        so resume-state handed out batch 2 as the new round's."""
        plan, phase_dir = _fixture_plan(tmp_path, phase)
        _fixture_status(phase_dir, round_id=None, state="validation_pending")
        _fixture_tasks(phase_dir, [1, 2], None)
        _write(phase_dir / "validation_batch_1.json", [])
        before = _classify(plan, phase)
        assert (before.case, before.action, before.batches) == ("b", "validate", (2,))

        claim = review_round.start_round(phase_dir)

        status = _status(phase_dir)
        assert (status["round_id"], status["state"]) == (claim.round_id, "models_running")
        after = _classify(plan, phase)
        assert (after.case, after.action, after.flags, after.ownership) == (
            "c", "rerun", ("--force",), "uncertain")


class TestAReportCutShortOrStaleIsNotHealed:
    def _uncompleted(self, plan: Path) -> None:
        state = StateManager(plan)
        state.state["phases_completed"].pop("code-review")
        state.save()

    def test_a_report_truncated_after_its_header(self, tmp_path):
        """Codex's repro: only the report's first two lines survive (a writer
        stopped mid-file), completion cleared. A plain review used to record
        the phase complete on an 80-byte report with no findings."""
        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        report = phase_dir / "report.md"
        report.write_text("\n".join(report.read_text(encoding="utf-8").split("\n")[:2]),
                          encoding="utf-8")
        self._uncompleted(plan)

        result = _run(runner, "code_review", plan)

        assert result.exit_code == 3, result.stdout
        assert "Auto-healed" not in result.stdout
        assert "not the whole report" in result.stdout
        assert "--reaggregate" in result.stdout
        assert "code-review" not in _completed(plan)
        _ok(runner.run_orchestrator("code_review", plan, "--reaggregate", "--yes", timeout=120))
        assert review_round.report_seal(phase_dir).whole

    def test_an_older_report_beside_validation_merged_again_in_the_same_round(
        self, tmp_path
    ):
        """A reaggregate re-merged validation.json and stopped before it
        rewrote the report: both record the round, but the report is not the
        one built from these verdicts."""
        runner, plan, phase_dir = _complete_round(
            tmp_path, "code_review", "code-review", CODE_ISSUES)
        validation = json.loads((phase_dir / "validation.json").read_text(encoding="utf-8"))
        validation["metadata"]["timestamp"] = "2026-11-08T12:00:00"
        (phase_dir / "validation.json").write_text(json.dumps(validation, indent=2),
                                                   encoding="utf-8")
        assert review_round.round_files(phase_dir).owner("validation.json") == "owned"
        self._uncompleted(plan)

        result = _run(runner, "code_review", plan)

        assert result.exit_code == 3, result.stdout
        assert "Auto-healed" not in result.stdout
        assert "--reaggregate" in result.stdout


class TestAnExplicitReaggregateKeepsToItsRound:
    @pytest.mark.parametrize("orchestrator, phase, response", PHASES)
    def test_another_round_s_files_are_refused(self, tmp_path, orchestrator, phase, response):
        """Codex's repro: R1 complete, a --force stopped right after stamping
        R2, then a hand-typed --reaggregate. It used to exit 0 with the phase
        complete, the task file and validation.json of R1 and a report
        claiming R2."""
        runner, plan, phase_dir = _complete_round(tmp_path, orchestrator, phase, response)
        first = _status(phase_dir)["round_id"]
        _interrupted_force_run(runner, orchestrator, plan)
        second = _status(phase_dir)["round_id"]
        kept = {name: (phase_dir / name).read_bytes()
                for name in ("validation_tasks.json", "validation.json", "report.md")}

        result = runner.run_orchestrator(orchestrator, plan, "--reaggregate", "--yes",
                                         timeout=120)

        assert result.exit_code == 3, result.stdout + result.stderr
        assert "--reaggregate refused" in result.stdout
        assert f"records round {first}" in result.stdout and second in result.stdout
        assert "--force" in result.stdout
        assert phase not in _completed(plan)
        assert {name: (phase_dir / name).read_bytes() for name in kept} == kept

    def test_the_shared_check_keeps_the_legacy_policy(self, tmp_path, capsys):
        claim = review_round.Claim(str(tmp_path), R1)
        _fixture_tasks(tmp_path, [1], None)
        _fixture_validation(tmp_path, None)
        review_round.refuse_foreign_round(claim, "retry")   # no id: content checks only
        _fixture_tasks(tmp_path, [1], R1)
        _fixture_validation(tmp_path, R1)
        review_round.refuse_foreign_round(claim, "retry")   # this round's own
        _fixture_validation(tmp_path, R2)
        with pytest.raises(SystemExit) as stop:
            review_round.refuse_foreign_round(claim, "the retry command")
        assert stop.value.code == 3
        out = capsys.readouterr().out
        assert "validation.json" in out and "the retry command" in out


# -- an earlier run's teardown not confirmed --------------------------------

_NO_CLEANUP = object()
_LEFT = "run-left-0005"
_NEW = "run-new-0006"


def _stopped_run(phase_dir: Path, cleanup: Any = _NO_CLEANUP, *, run_id: str = _LEFT) -> None:
    """A run its abort stopped, closed in `.status.json` with the `cleanup`
    the kill paths record (code health U77: `state` `pending`, `incomplete` or
    `complete`, beside the kill's `verdict`). Written as data: this tree has
    no writer of it yet."""
    run: Dict[str, Any] = {"run_id": run_id, "state": "aborted", "pid": 4194301,
                           "pgid": 4194301, "detached": True, "exit_code": 3,
                           "error": "aborted_by_request"}
    if cleanup is not _NO_CLEANUP:
        run["cleanup"] = cleanup
    phase_dir.mkdir(parents=True, exist_ok=True)
    _write(phase_dir / ".status.json", {"phase": phase_dir.name, "state": "running",
                                        "run": run})


def _kept_entry(plan: Path, cleanup: Any, *, run_id: str = _LEFT,
                phase_dir: Optional[Path] = None, **identity: Any) -> Path:
    """The registry entry the kill paths keep for such a run until its
    teardown is confirmed: carrying the same `cleanup` when its kill could
    not confirm every process stopped (U77's `run_registry.mark_cleanup`),
    none (`_NO_CLEANUP`) while an abort's kill is under way (U77's
    `RunHandle.finish` keeps it then), and the run's `phase_dir`."""
    from utils import run_registry

    root = run_registry.resolve_project_root(str(plan))
    path = Path(run_registry.entry_path(root, run_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    entry: Dict[str, Any] = {"run_id": run_id, "pid": 4194301, "pgid": 4194301,
                             "plan_file": str(plan), "mode": "review-code", **identity}
    if cleanup is not _NO_CLEANUP:
        entry["cleanup"] = cleanup
    if phase_dir is not None:
        entry["phase_dir"] = str(phase_dir)
    _write(path, entry)
    return path


_INCOMPLETE = {"state": "incomplete", "verdict": "unresolved", "pid": 4194301,
               "pgid": 4194301, "checked_at": "2026-11-08T22:35:36.432502"}

#: A string `state` other than `complete`: not confirmed.
_UNCONFIRMED_STATES = [
    pytest.param({"state": "pending", "verdict": None, "pid": 4194301, "pgid": 4194301,
                  "checked_at": "2026-11-08T22:35:30.000000"}, id="pending"),
    pytest.param(_INCOMPLETE, id="incomplete-unresolved"),
    pytest.param({"state": "incomplete", "verdict": "survived"}, id="incomplete-survived"),
    pytest.param({"state": "unresolved"}, id="unresolved"),
    pytest.param({"state": "a-state-added-later"}, id="unknown"),
]

#: Any other shape: not an object, or a `state` missing or not a string.
_MALFORMED = [
    pytest.param({"verdict": "killed"}, id="no-state"),
    pytest.param("incomplete", id="not-a-record"),
    pytest.param(["incomplete"], id="a-list"),
    pytest.param({"state": []}, id="list-state"),
    pytest.param({"state": {}}, id="dict-state"),
    pytest.param({"state": 3}, id="number-state"),
    pytest.param({"state": None}, id="null-state"),
]

_UNCONFIRMED = _UNCONFIRMED_STATES + _MALFORMED


def _refusal_names(err: str, cleanup: Any, record: Path, phase_dir: Path) -> None:
    """The refusal *err* names the run, the dir and the *record* that refused,
    and the cleanup that clears it: a malformed record as such."""
    malformed = not (isinstance(cleanup, dict) and isinstance(cleanup.get("state"), str))
    if malformed:
        # `status --abort` leaves a record it cannot read alone; the sweep
        # closes the run out from its entry.
        assert f"malformed cleanup record of run {_LEFT} in {phase_dir}" in err, err
        assert "--abort" not in err and f"--run-id {_LEFT}" in err
    else:
        assert f"the teardown of run {_LEFT} in {phase_dir} is not confirmed" in err, err
        assert f"--abort --run-id {_LEFT}" in err
    assert f"in {record}" in err, err
    assert "runs --sweep" in err
    assert "stopped before it changed anything" in err


_CONFIRMED = [
    pytest.param(_NO_CLEANUP, id="absent"),
    pytest.param(None, id="null"),
    pytest.param({"state": "complete", "verdict": "killed", "pid": 4194301,
                  "pgid": 4194301, "checked_at": "2026-11-08T22:36:00.000000"},
                 id="complete"),
]

_LANES = {
    "foreground": {},
    "launcher": {"detach": True},
    "launcher-force": {"detach": True, "force": True},
    "worker": {"internal_worker": True, "internal_run_id": _NEW},
    "held": {},
}


class TestAnUnconfirmedTeardownRefusesEveryEntry:
    """Codex round 5, point 2 (u81-r5/u77_retry_after_refusal.py): `--force`
    refused an abort whose teardown ended `incomplete`, but the abort had
    stopped the worker, so its lock was free; the next invocation took it
    without reading the cleanup record and ran while a child of the old run
    still wrote in the dir. Every entry now reads that record once it holds
    the lock and before its first write, and refuses unless it is absent or
    `complete`."""

    def _enter(self, phase_dir: Path, lane: str):
        from utils import detach

        run = detach.PhaseRun()
        run.take(phase_dir, mode="review-code", plan_file=str(phase_dir / "p.md"),
                 args=_args(**_LANES[lane]), phase="code-review")
        return run

    def _refused(self, phase_dir: Path, lane: str):
        from utils import detach, process_control

        outer = None
        if lane == "held":  # a composite step's or a front door's lock
            outer = process_control.PhaseLock(str(phase_dir / ".lock"))
            outer.acquire({"holder": "improve step", "pid": os.getpid()})
        try:
            with pytest.raises(SystemExit) as stop:
                with detach.PhaseRun() as run:
                    run.take(phase_dir, mode="review-code", plan_file=str(phase_dir / "p.md"),
                             args=_args(**_LANES[lane]), phase="code-review")
                    pytest.fail("the run went past its entry")
            if outer is not None:
                assert outer.held, "a held lock is its taker's to release"
        finally:
            if outer is not None:
                outer.release()
        assert stop.value.code == 3
        assert process_control.held_phase_lock(str(phase_dir / ".lock")) is None
        assert process_control.lock_is_free(str(phase_dir / ".lock"))

    @pytest.mark.parametrize("held_by", ["entry", "checkpoint"])
    @pytest.mark.parametrize("cleanup", _UNCONFIRMED)
    @pytest.mark.parametrize("lane", ["foreground", "launcher", "launcher-force", "worker",
                                      "held"])
    def test_every_entry_refuses_and_keeps_the_records(self, tmp_path, capsys, lane,
                                                        cleanup, held_by):
        """The run's entry is kept: carrying the cleanup itself, or none, so
        the checkpoint's value is what it backs."""
        _stopped_run(tmp_path, cleanup)
        entry = _kept_entry(tmp_path / "p.md", cleanup if held_by == "entry" else _NO_CLEANUP,
                            phase_dir=tmp_path)
        records = (tmp_path / ".status.json").read_bytes(), entry.read_bytes()
        self._refused(tmp_path, lane)
        assert ((tmp_path / ".status.json").read_bytes(), entry.read_bytes()) == records, \
            "records kept"
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            ".lock", ".multi-llm", ".status.json"]
        _refusal_names(capsys.readouterr().err, cleanup,
                       entry if held_by == "entry" else tmp_path / ".status.json", tmp_path)

    @pytest.mark.parametrize("kept", [False, True], ids=["no-entry", "entry-kept"])
    @pytest.mark.parametrize("cleanup", _CONFIRMED)
    @pytest.mark.parametrize("lane", ["foreground", "launcher", "worker", "held"])
    def test_no_record_or_a_confirmed_one_lets_every_entry_in(self, tmp_path, capsys, lane,
                                                               cleanup, kept):
        from utils import process_control

        _stopped_run(tmp_path, cleanup)
        if kept:
            _kept_entry(tmp_path / "p.md", cleanup, phase_dir=tmp_path)
        outer = None
        if lane == "held":
            outer = process_control.PhaseLock(str(tmp_path / ".lock"))
            outer.acquire({"holder": "improve step", "pid": os.getpid()})
        try:
            run = self._enter(tmp_path, lane)
            try:
                assert process_control.held_phase_lock(str(tmp_path / ".lock")) is not None
            finally:
                run.__exit__(None, None, None)
        finally:
            if outer is not None:
                outer.release()
        assert process_control.lock_is_free(str(tmp_path / ".lock"))
        assert "WARNING" not in capsys.readouterr().err

    @pytest.mark.parametrize("where", ["this-dir", "this-run"])
    def test_the_kept_registry_entry_refuses_once_the_record_was_replaced(
            self, tmp_path, capsys, where):
        """The phase record is another run's by now (no cleanup); the entry the
        kill paths kept for the stopped run still names this dir, or the run
        the record names."""
        if where == "this-dir":
            _stopped_run(tmp_path, run_id="run-next-0007")
            entry = _kept_entry(tmp_path / "p.md", _INCOMPLETE, phase_dir=tmp_path)
        else:
            _stopped_run(tmp_path)
            entry = _kept_entry(tmp_path / "p.md", _INCOMPLETE)  # no phase_dir in it
        kept = entry.read_bytes()
        self._refused(tmp_path, "foreground")
        assert entry.read_bytes() == kept
        err = capsys.readouterr().err
        assert f"the teardown of run {_LEFT} in" in err and str(entry) in err

    def test_an_entry_of_another_dir_or_a_confirmed_one_is_not_this_dir_s(self, tmp_path):
        from utils import detach

        _stopped_run(tmp_path, run_id="run-next-0007")
        _kept_entry(tmp_path / "p.md", _INCOMPLETE, phase_dir=tmp_path / "other")
        _kept_entry(tmp_path / "p.md", {"state": "complete", "verdict": "killed"},
                    run_id="run-done-0008", phase_dir=tmp_path)
        assert detach.unconfirmed_cleanup(str(tmp_path), plan_file=str(tmp_path / "p.md")) is None
        self._enter(tmp_path, "foreground").__exit__(None, None, None)

    def test_a_retry_after_the_refused_takeover_is_refused_too(
            self, tmp_path, capsys, monkeypatch):
        """Codex's sequence: `--force` stops the holder, whose abort ends
        `incomplete` (the record kept); the worker is gone and its lock free.
        The retry, with `--force` or without, and a foreground run all take
        the free lock and stop at the record."""
        from utils import detach, process_control

        lock_file = tmp_path / ".lock"
        holder = _hold_in_another_process(lock_file, run_id=_LEFT)

        def abort(phase_dir, **_kw):
            holder.communicate("\n", timeout=10)  # the worker is gone, and its lock
            status = _status(tmp_path)
            status["run"].update(state="aborted", exit_code=3, cleanup=_INCOMPLETE)
            _write(tmp_path / ".status.json", status)
            _kept_entry(tmp_path / "p.md", _INCOMPLETE, phase_dir=tmp_path)  # U77's abort
            return {"result": "incomplete", "verdict": "unresolved", "phase_dir": phase_dir}

        monkeypatch.setattr(detach, "abort_run", abort)
        try:
            with pytest.raises(SystemExit) as stop:
                self._enter(tmp_path, "launcher-force")
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
        assert stop.value.code == 3
        assert "not every process of that run is confirmed stopped" in capsys.readouterr().err
        assert process_control.lock_is_free(str(lock_file)), "the old worker's lock is gone"
        records = (tmp_path / ".status.json").read_bytes()
        for lane in ("launcher-force", "launcher", "foreground", "worker"):
            self._refused(tmp_path, lane)
            assert (tmp_path / ".status.json").read_bytes() == records, lane
            err = capsys.readouterr().err
            assert "is not confirmed (cleanup `incomplete`, verdict unresolved" in err

    def test_force_after_a_confirmed_stop_still_reads_the_record(
            self, tmp_path, capsys, monkeypatch):
        """The run `--force` stopped is confirmed gone, but an earlier run of
        this dir left a teardown its kept entry records as not confirmed."""
        from utils import detach, process_control

        lock_file = tmp_path / ".lock"
        entry = _kept_entry(tmp_path / "p.md", _INCOMPLETE, phase_dir=tmp_path)
        holder = _hold_in_another_process(lock_file, run_id="run-held-0009")

        def abort(phase_dir, **_kw):
            holder.communicate("\n", timeout=10)
            return {"result": "aborted", "phase_dir": phase_dir}

        monkeypatch.setattr(detach, "abort_run", abort)
        try:
            with pytest.raises(SystemExit) as stop:
                self._enter(tmp_path, "launcher-force")
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
        assert stop.value.code == 3
        assert process_control.held_phase_lock(str(lock_file)) is None
        assert process_control.lock_is_free(str(lock_file))
        assert entry.is_file()
        err = capsys.readouterr().err
        assert "stopping run run-held-0009" in err
        assert f"the teardown of run {_LEFT} in {tmp_path} is not confirmed" in err

    def test_a_refused_worker_tells_its_launch_why(self, tmp_path, capsys, monkeypatch):
        from utils import detach

        monkeypatch.setenv(detach.WORKER_ENV, "1")
        _stopped_run(tmp_path, _INCOMPLETE)
        _kept_entry(tmp_path / "p.md", _INCOMPLETE, phase_dir=tmp_path)
        self._refused(tmp_path, "worker")
        record = detach._read_worker_refusal(str(tmp_path), _NEW)
        assert record is not None and "is not confirmed (cleanup" in record["message"]

    def test_a_run_handle_refuses_before_it_claims_the_dir(self, tmp_path):
        """A mode with no `PhaseRun` entry: its worker's `RunHandle.start`."""
        from utils import detach, process_control, run_registry

        _stopped_run(tmp_path, _INCOMPLETE)
        _kept_entry(tmp_path / "p.md", _INCOMPLETE, phase_dir=tmp_path)
        records = (tmp_path / ".status.json").read_bytes()
        handle = detach.RunHandle(str(tmp_path), run_id=_NEW, mode="implement",
                                  plan_file=str(tmp_path / "p.md"), detached=False)
        with pytest.raises(detach.DetachError) as refused:
            handle.start()
        assert refused.value.exit_code == 3
        assert "is not confirmed (cleanup" in str(refused.value)
        assert (tmp_path / ".status.json").read_bytes() == records
        assert process_control.lock_is_free(str(tmp_path / ".lock"))
        assert not os.path.exists(run_registry.entry_path(handle.project_root, _NEW))

    def test_a_commit_front_door_refuses_before_its_first_write(self, tmp_path, capsys):
        from utils import commit_front_door as door
        from utils import process_control

        brief = tmp_path / "commits-feat.md"
        _phase_word, directory = door._lock_dir("review-commits", brief)
        _stopped_run(directory, _INCOMPLETE)
        _kept_entry(brief, _INCOMPLETE, phase_dir=directory)
        with pytest.raises(door.FrontDoorExit) as stop:
            door._take_lock("review-commits", "review-commits", brief)
        assert stop.value.exit_code == 3
        assert process_control.lock_is_free(str(directory / ".lock"))
        assert "is not confirmed (cleanup" in capsys.readouterr().err

    @pytest.mark.parametrize("mode, phase", [
        ("review-plan", "review-plan"),
        ("review-tasks", "review-tasks"),
        ("review-code", "code-review"),
    ])
    @pytest.mark.parametrize("lane", [[], ["--detach"], ["--detach", "--force"]],
                             ids=["foreground", "detach", "detach-force"])
    def test_a_review_run_changes_nothing(self, tmp_path, mode, phase, lane):
        runner = SkillRunner(tmp_path / "harness")
        runner.set_config({"response": [SUGGESTION]})
        plan = _plan(tmp_path)
        if mode == "review-code":
            _implemented(plan)
        phase_dir = _phase(plan, phase)
        _stopped_run(phase_dir, _INCOMPLETE)
        _kept_entry(plan, _INCOMPLETE, phase_dir=phase_dir)
        before = _snapshot(plan, skip=(".lock",))
        result = runner._run_command(
            [sys.executable, str(SKILL_ROOT / "multi_llm.py"), mode, str(plan),
             "--models", MODEL, *lane], plan, 60, False)
        assert _snapshot(plan, skip=(".lock",)) == before
        assert result.exit_code == 3, result.stdout + result.stderr
        assert "is not confirmed (cleanup" in result.stdout + result.stderr


#: A pid above any `pid_max`: the run's worker is gone.
_GONE_PID = 2147483647
_OTHER = "<another phase dir>"

#: Registry entries of another dir and another run, malformed in each field
#: the guard could read, the file itself included.
_FOREIGN_ENTRIES = [
    *[pytest.param({"run_id": "run-else-0010", "phase_dir": _OTHER, "cleanup": shape.values[0]},
                   id=f"cleanup-{shape.id}") for shape in _UNCONFIRMED],
    pytest.param({"run_id": "run-else-0010", "phase_dir": ["a", "list"],
                  "cleanup": {"state": []}}, id="phase-dir-a-list"),
    pytest.param({"run_id": "run-else-0010", "phase_dir": "nul\u0000byte",
                  "cleanup": {"state": {}}}, id="phase-dir-nul"),
    pytest.param({"run_id": {"not": "a string"}, "phase_dir": 7, "cleanup": ["x"]},
                 id="ids-not-strings"),
    pytest.param({"cleanup": {"state": []}}, id="no-ids"),
    pytest.param("{not json", id="not-json"),
    pytest.param("[1, 2]", id="not-an-object"),
]


class TestTheRegistryDecidesAndAMalformedRecordRefuses:
    """Codex round 6 (u81-r6/malformed_cleanup.py). Point 1: a checkpoint
    cleanup that neither `util runs --sweep` nor `status --abort` rewrites (a
    malformed one, a state they do not retry) refused every entry for good,
    after the sweep had confirmed the run gone and dropped its registry
    entry. The registry decides: the checkpoint's value counts only while
    the run's entry is kept, and is stale (a warning) once it is gone. Point
    2: a `state` that is a list or an object raised `TypeError` past the
    exit-3 refusal and kept the lock taken, and so did a malformed entry of
    another dir."""

    _enter = TestAnUnconfirmedTeardownRefusesEveryEntry._enter
    _refused = TestAnUnconfirmedTeardownRefusesEveryEntry._refused

    def _entered(self, phase_dir: Path, lane: str) -> None:
        from utils import process_control

        outer = None
        if lane == "held":
            outer = process_control.PhaseLock(str(phase_dir / ".lock"))
            outer.acquire({"holder": "improve step", "pid": os.getpid()})
        try:
            run = self._enter(phase_dir, lane)
            try:
                assert process_control.held_phase_lock(str(phase_dir / ".lock")) is not None
            finally:
                run.__exit__(None, None, None)
        finally:
            if outer is not None:
                outer.release()
        assert process_control.lock_is_free(str(phase_dir / ".lock"))

    @staticmethod
    def _stale_warning(err: str, phase_dir: Path) -> None:
        assert "ERROR" not in err, err
        warnings = [line for line in err.splitlines() if "is stale" in line]
        assert len(warnings) == 1, err
        assert warnings[0].startswith("WARNING: the cleanup state ")
        assert f"of run {_LEFT} recorded in {phase_dir / '.status.json'} is stale" in warnings[0]

    @pytest.mark.parametrize("cleanup", _UNCONFIRMED)
    @pytest.mark.parametrize("lane", ["foreground", "launcher", "worker", "held"])
    def test_a_checkpoint_value_no_entry_keeps_is_stale_and_goes_on(
            self, tmp_path, capsys, lane, cleanup):
        _stopped_run(tmp_path, cleanup)
        records = (tmp_path / ".status.json").read_bytes()
        self._entered(tmp_path, lane)
        assert (tmp_path / ".status.json").read_bytes() == records
        self._stale_warning(capsys.readouterr().err, tmp_path)

    @pytest.mark.parametrize("cleanup", [
        pytest.param("incomplete", id="string"),
        pytest.param({"verdict": "killed"}, id="no-state"),
        pytest.param({"state": "unresolved"}, id="unresolved"),
        pytest.param({"state": []}, id="list-state"),
        pytest.param({"state": {}}, id="dict-state"),
    ])
    def test_the_sweep_that_drops_the_entry_lifts_the_refusal(self, tmp_path, capsys, cleanup):
        """Codex's sequence: the run's entry carries what its checkpoint does;
        the sweep confirms the worker gone and drops the entry, leaving the
        checkpoint's value as it was. Every entry then goes on."""
        from utils import run_registry

        _stopped_run(tmp_path, cleanup)
        entry = _kept_entry(tmp_path / "p.md", cleanup, phase_dir=tmp_path, pid=_GONE_PID,
                            pgid=None)
        self._refused(tmp_path, "foreground")
        _refusal_names(capsys.readouterr().err, cleanup, entry, tmp_path)
        rows = run_registry.sweep(run_registry.resolve_project_root(str(tmp_path / "p.md")))
        assert [(row["run_id"], row["action"]) for row in rows] == [(_LEFT, "closed")]
        assert not entry.exists()
        assert _status(tmp_path)["run"]["cleanup"] == cleanup, "the checkpoint's value stays"
        for lane in ("foreground", "launcher", "worker", "held"):
            self._entered(tmp_path, lane)
            self._stale_warning(capsys.readouterr().err, tmp_path)

    @pytest.mark.parametrize("foreign", _FOREIGN_ENTRIES)
    def test_another_dir_s_entry_is_never_read(self, tmp_path, capsys, foreign):
        from utils import detach, run_registry

        _stopped_run(tmp_path, run_id="run-next-0007")
        if isinstance(foreign, dict) and foreign.get("phase_dir") == _OTHER:
            foreign = dict(foreign, phase_dir=str(tmp_path / "other"))
        path = Path(run_registry.entry_path(
            run_registry.resolve_project_root(str(tmp_path / "p.md")), "run-else-0010"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(foreign if isinstance(foreign, str) else json.dumps(foreign),
                        encoding="utf-8")
        assert detach.unconfirmed_cleanup(str(tmp_path), plan_file=str(tmp_path / "p.md")) is None
        for lane in ("foreground", "launcher", "worker", "held"):
            self._entered(tmp_path, lane)
        assert capsys.readouterr().err == ""
        # Beside it, this dir's own entry still refuses.
        own = _kept_entry(tmp_path / "p.md", _INCOMPLETE, phase_dir=tmp_path)
        self._refused(tmp_path, "foreground")
        _refusal_names(capsys.readouterr().err, _INCOMPLETE, own, tmp_path)

    @pytest.mark.parametrize("lane", ["foreground", "launcher", "launcher-force", "worker",
                                      "held"])
    def test_a_read_that_raises_refuses_and_frees_the_lock_it_took(
            self, tmp_path, capsys, monkeypatch, lane):
        """`_refused` checks exit 3, the lock free, and a held lock still its
        taker's."""
        from utils import detach

        def broken(*_args, **_kwargs):
            raise RuntimeError("the record broke the read")

        monkeypatch.setattr(detach, "_cleanup_records", broken)
        _stopped_run(tmp_path, _INCOMPLETE)
        records = (tmp_path / ".status.json").read_bytes()
        self._refused(tmp_path, lane)
        assert (tmp_path / ".status.json").read_bytes() == records
        err = capsys.readouterr().err
        assert f"malformed cleanup record of an earlier run in {tmp_path}" in err, err
        assert "reading it failed: RuntimeError: the record broke the read" in err
        assert str(tmp_path / ".status.json") in err
        assert os.path.join(str(tmp_path), ".multi-llm", "runs") in err
        assert "runs --sweep" in err and "stopped before it changed anything" in err

    def test_a_run_handle_and_a_commit_front_door_free_their_lock_too(
            self, tmp_path, capsys, monkeypatch):
        from utils import commit_front_door as door
        from utils import detach, process_control

        def broken(*_args, **_kwargs):
            raise RuntimeError("the record broke the read")

        monkeypatch.setattr(detach, "_cleanup_records", broken)
        handle = detach.RunHandle(str(tmp_path), run_id=_NEW, mode="implement",
                                  plan_file=str(tmp_path / "p.md"), detached=False)
        with pytest.raises(detach.DetachError) as refused:
            handle.start()
        assert refused.value.exit_code == 3
        assert "malformed cleanup record" in str(refused.value)
        assert process_control.lock_is_free(str(tmp_path / ".lock"))
        brief = tmp_path / "commits-feat.md"
        _phase_word, directory = door._lock_dir("review-commits", brief)
        directory.mkdir(parents=True, exist_ok=True)
        with pytest.raises(door.FrontDoorExit) as stop:
            door._take_lock("review-commits", "review-commits", brief)
        assert stop.value.exit_code == 3
        assert process_control.lock_is_free(str(directory / ".lock"))
        assert "malformed cleanup record" in capsys.readouterr().err

    @pytest.mark.parametrize("cleanup", _MALFORMED)
    def test_the_refusal_names_the_malformed_record(self, tmp_path, cleanup):
        from utils import detach

        _stopped_run(tmp_path)
        entry = _kept_entry(tmp_path / "p.md", cleanup, phase_dir=tmp_path)
        found = detach.unconfirmed_cleanup(str(tmp_path), plan_file=str(tmp_path / "p.md"))
        assert found == {"run_id": _LEFT, "cleanup": cleanup, "record": str(entry),
                         "problem": detach.CLEANUP_MALFORMED}
