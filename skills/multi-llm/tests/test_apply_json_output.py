"""``--output-format json`` on the three apply phases: stdout is one JSON document.

Every path drives the real command (``<orchestrator>.main(argv)`` or
``multi_llm.main(argv)``) and parses the whole captured stdout, so a marker,
an ``[AUTO_RESOLVED]`` line or a progress line printed anywhere in the run
fails the test. The contract and the dispatcher's result line that follows the
document are in docs/development/dispatcher.md ("Entrypoint and result
protocol", **JSON output**).
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import apply_code_fixes_orchestrator  # noqa: E402
import apply_suggestions_orchestrator  # noqa: E402
import apply_task_suggestions_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, detach  # noqa: E402
from utils.apply_orchestrator_base import OrchestratorError  # noqa: E402
from utils.state_manager import StateManager, stamp_stable_ids  # noqa: E402

# phase -> (module, review dir, phase dir, phases recorded completed before it)
PHASES = {
    "apply-suggestions": (
        apply_suggestions_orchestrator, "review-plan", "apply-suggestions",
        ("review-plan",),
    ),
    "apply-task-suggestions": (
        apply_task_suggestions_orchestrator, "review-tasks", "apply-task-suggestions",
        ("review-plan", "apply-suggestions", "generate-tasks", "review-tasks"),
    ),
    "apply-code-fixes": (
        apply_code_fixes_orchestrator, "code-review", "apply-fixes",
        ("review-plan", "apply-suggestions", "generate-tasks", "implement",
         "code-review"),
    ),
}
PHASE_IDS = list(PHASES)

PLAN = """# Plan

## Overview
Ship the notification service.

## T001: Notification model
Create the model.
"""

TASKS = """# Tasks

## T001: Notification model
- Depends on: none
"""


def _suggestion(title, importance, reference):
    return {
        "title": title, "desc": f"{title}.", "type": "modification",
        "importance": importance, "reference": reference, "file": "src/app.py",
        "source_model": "mock",
    }


GROUPS = [
    {"theme": "Clarify the model", "category": "clarity", "models": ["mock"],
     "suggestions": [_suggestion("Clarify the model", "HIGH", "T001")]},
    {"theme": "Ambiguous scope", "category": "clarity", "models": ["mock"],
     "suggestions": [_suggestion("Ambiguous scope", "MEDIUM", "Overview")]},
    {"theme": "Out of scope", "category": "testing", "models": ["mock"],
     "suggestions": [_suggestion("Out of scope", "LOW", "Overview")]},
]

VALIDATION = {
    "metadata": {"schema_version": "2.0", "model": "mock"},
    "groups": [
        {"group_index": 0, "status": "valid", "reason": "Clear.", "confidence": 0.9},
        {"group_index": 1, "status": "needs-human-decision", "reason": "Unclear scope.",
         "confidence": 0.5, "error_type": "real_ambiguity", "recoverable": False},
        {"group_index": 2, "status": "invalid", "reason": "Out of scope.",
         "confidence": 0.8},
    ],
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository holding a plan, its tasks file, a source file and cwd."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    plan = root / "plans" / "p.md"
    plan.parent.mkdir()
    plan.write_text(PLAN, encoding="utf-8")
    (root / "plans" / "p" / "tasks").mkdir(parents=True)
    (root / "plans" / "p" / "tasks" / "tasks.md").write_text(TASKS, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"],
        cwd=root, check=True,
    )
    monkeypatch.chdir(root)
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    return plan


def _ready(plan: Path, phase: str, validation=VALIDATION) -> Path:
    """The review artifacts and state *phase* needs; its phase dir."""
    _module, review, phase_dir, done = PHASES[phase]
    review_dir = plan.parent / "p" / review
    review_dir.mkdir(parents=True, exist_ok=True)
    groups = copy.deepcopy(GROUPS)
    stamp_stable_ids(groups)
    (review_dir / "grouped.json").write_text(
        json.dumps({"format_version": 2, "groups": groups}), encoding="utf-8")
    (review_dir / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    state = StateManager(plan)
    for word in done:
        state.mark_phase_completed(word)
    state.save()
    return plan.parent / "p" / phase_dir


def _exit_code(exc: SystemExit) -> int:
    return 0 if exc.code is None else exc.code if isinstance(exc.code, int) else 1


def _run(phase, plan, capsys, *flags):
    """``<phase orchestrator>.main`` under ``--output-format json``: (exit, out, err)."""
    module = PHASES[phase][0]
    try:
        code = module.main(["--plan-file", str(plan), "--output-format", "json", *flags])
    except SystemExit as exc:
        code = _exit_code(exc)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _written(phase_dir: Path):
    return json.loads((phase_dir / "orchestrator_output.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The paths that print a payload: the payload, and nothing else
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phase", PHASE_IDS)
def test_yes_sends_its_auto_resolved_lines_to_stderr(repo, capsys, phase):
    phase_dir = _ready(repo, phase)
    code, out, err = _run(phase, repo, capsys, "--yes")
    assert code == 0, err
    assert json.loads(out) == _written(phase_dir)
    # Row 10 is announced in the constructor, before `run()`; row 11 inside it.
    assert "[AUTO_RESOLVED] needs-human-decision:" in err
    assert "[AUTO_RESOLVED] no-selections-confirmation:" in err


@pytest.mark.parametrize("phase", PHASE_IDS)
def test_agent_decide_sends_its_judging_markers_to_stderr(repo, capsys, phase):
    phase_dir = _ready(repo, phase)
    code, out, err = _run(phase, repo, capsys, "--agent-decide", "--no-confirm")
    assert code == 0, err
    assert json.loads(out) == _written(phase_dir)
    assert "[HUMAN_DECISION_JUDGING_PENDING]" in err
    assert "Items to judge: 1" in err


@pytest.mark.parametrize("phase", PHASE_IDS)
def test_dry_run_subagents_sends_its_preflight_to_stderr(repo, capsys, phase):
    phase_dir = _ready(repo, phase)
    code, out, err = _run(phase, repo, capsys, "--dry-run-subagents", "--yes")
    assert code == 0, err
    assert json.loads(out) == _written(phase_dir)
    assert "[write-subagents] materialized" in err


@pytest.mark.parametrize("phase", PHASE_IDS)
def test_text_mode_still_prints_the_plan_of_work(repo, capsys, phase):
    _ready(repo, phase)
    module = PHASES[phase][0]
    assert module.main(["--plan-file", str(repo), "--no-confirm"]) == 0
    out = capsys.readouterr().out
    assert "TO APPLY (BATCHED)" in out


# ---------------------------------------------------------------------------
# The exits that print no payload of their own: a fresh stop object
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phase", PHASE_IDS)
def test_a_dry_run_prints_a_fresh_dry_run_object(repo, capsys, phase):
    phase_dir = _ready(repo, phase)
    phase_dir.mkdir(parents=True, exist_ok=True)
    stale = {"status": "an earlier run's output", "batches": ["stale"]}
    (phase_dir / "orchestrator_output.json").write_text(json.dumps(stale), encoding="utf-8")

    code, out, err = _run(phase, repo, capsys, "--dry-run", "--no-confirm")

    assert code == 0, err
    document = json.loads(out)
    assert document["status"] == "dry_run"
    assert document["exit_code"] == 0
    assert document["phase"] == phase
    assert [item["title"] for batch in document["batches"] for item in batch["items"]] \
        == ["Clarify the model"]
    assert document["summary"] == {
        "batch_count": 1, "valid_count": 1, "needs_human_count": 1, "skipped_count": 1,
        "user_skipped_count": 0,
    }
    # The selection, under the output file's own keys.
    assert [row["title"] for row in document["to_apply"]] == ["Clarify the model"]
    assert [row["title"] for row in document["needs_human_review"]] == ["Ambiguous scope"]
    assert [row["title"] for row in document["skipped_items"]] == ["Out of scope"]
    # Fresh, and the dry run wrote nothing over the earlier file.
    assert "stale" not in out
    assert _written(phase_dir) == stale
    assert "--- DRY RUN ---" in err


@pytest.mark.parametrize("phase", PHASE_IDS)
def test_a_missing_plan_prints_an_error_object(repo, capsys, phase):
    missing = repo.parent / "absent.md"
    code, out, err = _run(phase, missing, capsys, "--yes")
    assert code == 1
    document = json.loads(out)
    assert document["status"] == "error"
    assert document["exit_code"] == 1
    assert "not found" in document["message"]
    assert "not found" in err


@pytest.mark.parametrize(
    ("exit_code", "status"), [(2, "nothing_to_do"), (3, "needs_human"), (1, "error")]
)
def test_a_stop_names_its_exit(repo, capsys, monkeypatch, exit_code, status):
    phase = "apply-task-suggestions"
    _ready(repo, phase)

    def held(self):
        raise OrchestratorError("held for a reason", exit_code=exit_code)

    monkeypatch.setattr(
        apply_task_suggestions_orchestrator.ApplyTaskSuggestionsOrchestrator,
        "_load_and_parse_inputs", held,
    )
    code, out, _err = _run(phase, repo, capsys, "--no-confirm")
    assert code == exit_code
    assert json.loads(out) == {
        "status": status, "phase": phase, "exit_code": exit_code,
        "plan_file": str(repo), "message": "held for a reason",
    }


def test_a_revalidation_handed_to_the_host_names_its_task_file(repo, capsys):
    phase = "apply-suggestions"
    validation = copy.deepcopy(VALIDATION)
    validation["groups"][1].update(status="validation_failed", error_type="timeout",
                                   recoverable=True)
    _ready(repo, phase, validation)
    code, out, err = _run(phase, repo, capsys, "--revalidate", "--no-confirm")
    assert code == 0, err
    document = json.loads(out)
    assert document["status"] == "revalidation_pending"
    assert Path(document["revalidation_tasks"]).is_file()
    assert "[REVALIDATION_PENDING]" in err


# ---------------------------------------------------------------------------
# Through the dispatcher: the document, then the result line hosts parse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phase", PHASE_IDS)
@pytest.mark.parametrize("flags", [("--no-confirm",), ("--dry-run", "--yes")],
                         ids=["emit", "dry-run"])
def test_the_dispatcher_prints_the_document_then_its_result_line(
    repo, capsys, phase, flags
):
    phase_dir = _ready(repo, phase)
    code = multi_llm.main([phase, str(repo), "--output-format", "json", *flags])
    out = capsys.readouterr().out

    assert code == 0
    lines = out.splitlines()
    assert lines[-1].startswith(cli_common.RESULT_MARKER)
    assert sum(line.startswith(cli_common.RESULT_MARKER) for line in lines) == 1
    document = json.loads(out.rsplit(cli_common.RESULT_MARKER, 1)[0])
    if "--dry-run" in flags:
        assert document["status"] == "dry_run"
    else:
        assert document == _written(phase_dir)
    assert cli_common.parse_result_line(out)["exit"] == 0
