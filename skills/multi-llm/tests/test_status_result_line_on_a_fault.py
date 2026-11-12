"""What a `status` poll's result line keeps when the poll fails partway.

The dispatcher prints ``MULTI_LLM_RESULT`` on a crash or an interrupt too
(exit 1), carrying whatever the command recorded before the fault. ``status``
records each field as soon as it knows it
(docs/development/status.md#how-status-is-built): a fault after the pinned
run was read keeps its ``mode``, ``phase``, ``progress``, ``progress_lines`` and
stop line, a fault in the render keeps everything decided before it, and a
fault before a field is recorded leaves that field off.

Each case raises an ``OSError`` or a ``KeyboardInterrupt`` at one call on a
boundary between two of ``main``'s stages, between two of the records inside
one, or inside ``render_run_text``, and pins the exit code and which of the
clean poll's fields the line keeps (:data:`FAULTS`, against :data:`CLEAN`).
"""

from __future__ import annotations

import json
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import multi_llm  # noqa: E402
from utils import (  # noqa: E402
    cli_common,
    detach,
    process_control,
    run_status,
    status_commands,
    status_file,
    status_native,
    status_text,
    status_views,
)
from utils.output_handler import phase_dir_path  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

#: Where each fault point's function lives (the module whose binding the
#: stage calls).
OWNERS = {
    "run_status": run_status,
    "status_views": status_views,
    "status_text": status_text,
    "status_native": status_native,
    "status_commands": status_commands,
    "detach": detach,
}


# ---------------------------------------------------------------------------
# The polls
# ---------------------------------------------------------------------------

def _record(plan: Path, phase_name: str, run_id: str, **run: Any) -> Dict[str, Any]:
    phase = phase_dir_path(plan, phase_name)
    return {
        "run_id": run_id, "mode": phase_name, "pid": 987654321,
        "process_start_time": None, "started_at": "2026-11-08T19:59:00",
        "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 3600.0,
        "host": "claude-code", "plan_file": str(plan),
        "project_root": str(plan.parent), "phase_dir": str(phase),
        "detached": True, "log_path": str(phase / "run.log"), **run,
    }


def _write(plan: Path, phase_name: str, run: Dict[str, Any], **status: Any) -> None:
    phase = phase_dir_path(plan, phase_name)
    phase.mkdir(parents=True, exist_ok=True)
    status_file.write_status(str(phase), {"phase": phase_name, "run": run, **status})


def build_failed(plan: Path) -> None:
    """A failed review-plan run that left its progress and its own stop line."""
    phase = phase_dir_path(plan, "review-plan")
    _write(plan, "review-plan", _record(
        plan, "review-plan", "review-run-1", state="failed", exit_code=3,
        error="Stop reason from worker"),
        state="models_running",
        progress={"step": "fanout", "completed": 0, "total": 1, "models": {}},
        output_files={"report": str(phase / "report.md")},
        validation_counts={"valid": 1, "invalid": 0, "needs_human_decision": 0,
                           "validation_failed": 0})


def build_partial(plan: Path) -> None:
    """A finished review-plan run that exited 0 as ``partial``, with every
    field a pinned poll re-records from its run."""
    phase = phase_dir_path(plan, "review-plan")
    _write(plan, "review-plan", _record(
        plan, "review-plan", "review-run-2", state="done", exit_code=0,
        result_state="partial", plan_arg="plan.md"),
        state="complete",
        progress={"step": "done", "completed": 2, "total": 2, "models": {
            "a:m1": {"state": "done"},
            "b:m2": {"state": "failed", "error": "boom"}}},
        output_files={"report": str(phase / "report.md")},
        consolidation_recommended={"valid_groups": 9, "threshold": 5,
                                   "command": "--consolidate"},
        validation_counts={"valid": 2, "invalid": 1, "needs_human_decision": 0,
                           "validation_failed": 0},
        high_priority_groups={"count": 2},
        validation_context={"mode": "truncated", "phase": "review-plan", "sources": [{
            "label": "plan", "mode": "truncated", "chars": 100, "bytes": 5000,
            "path": str(plan), "sha256": "0" * 64, "dropped_sections": ["Later"],
            "reason": "over the limit"}]})
    # The phase is complete in the workflow, so the poll names the next one.
    state = StateManager(plan)
    state.mark_phase_completed("review-plan")
    state.save()


def build_listing(plan: Path) -> None:
    """Two runs and no selector: one still running, one finished."""
    _write(plan, "generate-tasks", _record(
        plan, "generate-tasks", "gen-run-1", state="running"),
        state="models_running",
        progress={"step": "fanout", "completed": 1, "total": 3, "models": {}})
    _write(plan, "review-plan", _record(
        plan, "review-plan", "review-run-3", state="done", exit_code=0),
        state="complete",
        output_files={"report": str(phase_dir_path(plan, "review-plan") / "report.md")})


#: name -> (fixture, the poll's arguments after the plan).
SCENARIOS = {
    "failed": (build_failed, ["--view", "run", "--phase", "review-plan"]),
    "partial": (build_partial, ["--view", "run", "--phase", "review-plan"]),
    "partial-json": (build_partial, ["--view", "run", "--phase", "review-plan", "--json"]),
    "listing": (build_listing, ["--exit-zero"]),
}


def make_plan(root: Path) -> Path:
    """A committed plan in a git repository at *root* (the workflow view
    reads its HEAD)."""
    plan = root / "plan.md"
    plan.write_text("# Plan\n\n## Step 1\n\nBody.\n", encoding="utf-8")
    for argv in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.name=t", "-c", "user.email=t@example.invalid",
                  "commit", "-q", "-m", "plan"]):
        subprocess.run(["git", *argv], cwd=root, check=True, capture_output=True)
    return plan


def result_payload(out: str, root: Path) -> Dict[str, Any]:
    """The poll's ``MULTI_LLM_RESULT`` object, with *root* and the skill
    directory spelled ``<root>`` and ``<skill>``, and ``/`` between path parts.

    A ``next`` command is spelled argument by argument (``shlex.split``): it
    is encoded for the POSIX shell (``cli_common._quote``), which leaves a
    path bare where it needs no quoting and single-quotes a Windows one,
    whose backslashes it must not leave as escapes. What must hold is the
    arguments it passes, whatever quoting a path needed."""
    line = [ln for ln in out.splitlines() if ln.startswith(cli_common.RESULT_MARKER)][-1]
    payload = json.loads(line[len(cli_common.RESULT_MARKER):])
    skill = str(cli_common.skill_dir())

    def spell(value: Any) -> Any:
        if isinstance(value, str):
            return (value.replace(str(root), "<root>").replace(skill, "<skill>")
                    .replace("\\", "/"))
        if isinstance(value, list):
            return [spell(item) for item in value]
        if isinstance(value, dict):
            return {key: spell(item) for key, item in value.items()}
        return value

    spelled = spell(payload)
    if isinstance(payload.get("next"), str):
        spelled["next"] = " ".join(spell(arg) for arg in shlex.split(payload["next"]))
    return spelled


#: The exit-1 line of a poll that recorded nothing (the dispatcher's own).
BARE = {"mode": "status", "phase": None, "state": "error", "exit": 1,
        "artifacts": [], "next": None}


def expected(scenario: str, kept: List[str]) -> Dict[str, Any]:
    """:data:`BARE` with the clean poll's value of each *kept* field."""
    return {**BARE, **{key: CLEAN[scenario][key] for key in kept}}


# ---------------------------------------------------------------------------
# What each poll's line holds, clean and at each fault
# ---------------------------------------------------------------------------

CLEAN_LINE = "clean line"

#: Each poll's clean result line.
CLEAN: Dict[str, Dict[str, Any]] = {
    "failed": {
        "mode": "review-plan",
        "phase": "review-plan",
        "state": "needs-human",
        "exit": 3,
        "artifacts": [
            "<root>/plan/review-plan/report.md",
            "<root>/plan/review-plan/run.log",
        ],
        "next": None,
        "validation_counts": {
            "valid": 1,
            "invalid": 0,
            "needs_human_decision": 0,
            "validation_failed": 0,
        },
        "progress": {"step": "fanout", "completed": 0, "total": 1, "models": {}},
        "progress_lines": ["fanout"],
        "error": "Stop reason from worker",
    },
    "partial": {
        "mode": "review-plan",
        "phase": "review-plan",
        "state": "partial",
        "exit": 0,
        "artifacts": [
            "<root>/plan/review-plan/report.md",
            "<root>/plan/review-plan/run.log",
        ],
        "next": None,
        "consolidation_recommended": {
            "valid_groups": 9, "threshold": 5, "command": "--consolidate"},
        "high_priority_groups": {"count": 2},
        "validation_context": {
            "mode": "truncated",
            "phase": "review-plan",
            "sources": [
                {
                    "label": "plan",
                    "mode": "truncated",
                    "chars": 100,
                    "bytes": 5000,
                    "path": "<root>/plan.md",
                    "sha256": "0" * 64,
                    "dropped_sections": ["Later"],
                    "reason": "over the limit",
                },
            ],
        },
        "validation_counts": {
            "valid": 2,
            "invalid": 1,
            "needs_human_decision": 0,
            "validation_failed": 0,
        },
        "notice": "Validation saw 100 of 5000 chars of the plan",
        "progress": {
            "step": "done",
            "completed": 2,
            "total": 2,
            "models": {
                "a:m1": {"state": "done"},
                "b:m2": {"state": "failed", "error": "boom"},
            },
        },
        "progress_lines": ["[PROGRESS] fan-out over 2 models: a:m1 (?), b:m2 (?)", "done"],
        "user_next_step": {
            "phase": "apply-suggestions",
            "command": "/multi-llm:multi-llm --apply-suggestions plan.md",
            "complete": False,
            "markdown": "```\n/multi-llm:multi-llm --apply-suggestions plan.md\n```",
        },
    },
    "listing": {
        "mode": "status",
        "phase": None,
        "state": "running",
        "exit": 0,
        "artifacts": [
            "<root>/plan/generate-tasks/run.log",
            "<root>/plan/review-plan/report.md",
            "<root>/plan/review-plan/run.log",
        ],
        "next": (
            "uv run --frozen --no-dev --project <skill> -- python <skill>/multi_llm.py "
            "status <root>/plan.md --view run --phase generate-tasks --run-id gen-run-1 "
            "--host claude-code --wait"),
    },
}
CLEAN["partial-json"] = CLEAN["partial"]

#: ``(scenario, owner, function, call number, kept[, on OSError])``: the fault
#: is raised at that call of ``owner.function``. *kept* is the fields of the
#: clean line that the exit-1 line holds, for a KeyboardInterrupt and, unless
#: the sixth item says otherwise, for an OSError. ``CLEAN_LINE`` there: an
#: advisory guard swallows the OSError and the poll answers as if nothing
#: failed. The points, in call order: the gather (``_discovery_report``, the
#: superseded runs), the decide (its first record, then each read between its
#: records), the render (``user_speller`` opens it), and the records after it.
FAULTS: List[tuple] = [
    ("failed", "run_status", "_discovery_report", 1,
     []),
    ("failed", "status_native", "set_aside_superseded_runs", 1,
     [], CLEAN_LINE),
    ("failed", "run_status", "record_mode_phase", 1,
     []),
    ("failed", "status_text", "_progress_lines", 1,
     ["mode", "phase", "progress"]),
    ("failed", "status_views", "exit_code_for", 1,
     ["mode", "phase", "progress", "progress_lines", "error"]),
    ("failed", "run_status", "_commit_fixes_view", 1,
     ["mode", "phase", "progress", "progress_lines", "error"]),
    ("failed", "status_views", "_workflow_payload", 1,
     ["mode", "phase", "progress", "progress_lines", "error"], CLEAN_LINE),
    ("failed", "status_native", "state_apply_views", 1,
     ["mode", "phase", "progress", "progress_lines", "error"], CLEAN_LINE),
    ("failed", "status_commands", "user_speller", 1,
     ["mode", "phase", "progress", "progress_lines", "error"]),
    ("failed", "status_text", "render_run_text", 1,
     ["mode", "phase", "progress", "progress_lines", "error"]),
    ("failed", "status_commands", "render_runner_text", 1,
     ["mode", "phase", "progress", "progress_lines", "error"]),
    ("failed", "status_text", "_validation_context_notice", 1,
     ["mode", "phase", "progress", "progress_lines", "error"]),
    ("failed", "run_status", "record_artifact", 1,
     ["mode", "phase", "progress", "progress_lines", "error"]),
    ("failed", "run_status", "record_validation_counts", 1,
     ["mode", "phase", "artifacts", "progress", "progress_lines", "error"]),
    ("partial", "run_status", "_discovery_report", 1,
     []),
    ("partial", "status_native", "set_aside_superseded_runs", 1,
     [], CLEAN_LINE),
    ("partial", "run_status", "record_mode_phase", 1,
     []),
    ("partial", "status_text", "_progress_lines", 1,
     ["mode", "phase", "progress"]),
    ("partial", "status_views", "exit_code_for", 1,
     ["mode", "phase", "progress", "progress_lines"]),
    ("partial", "status_views", "_result_state", 1,
     ["mode", "phase", "progress", "progress_lines"]),
    ("partial", "run_status", "_commit_fixes_view", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial", "status_views", "_workflow_payload", 1,
     ["mode", "phase", "state", "progress", "progress_lines"], CLEAN_LINE),
    ("partial", "status_native", "state_apply_views", 1,
     ["mode", "phase", "state", "progress", "progress_lines"], CLEAN_LINE),
    ("partial", "run_status", "_emit_user_next_step", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial", "status_commands", "user_speller", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial", "status_text", "render_run_text", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial", "status_commands", "render_runner_text", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial", "status_text", "_validation_context_notice", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial", "run_status", "record_notice", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial", "run_status", "record_artifact", 1,
     ["mode", "phase", "state", "notice", "progress", "progress_lines"]),
    ("partial", "run_status", "record_consolidation_recommended", 1,
     [
         "mode",
         "phase",
         "state",
         "artifacts",
         "notice",
         "progress",
         "progress_lines",
     ]),
    ("partial", "run_status", "record_validation_context", 1,
     [
         "mode",
         "phase",
         "state",
         "artifacts",
         "consolidation_recommended",
         "notice",
         "progress",
         "progress_lines",
     ]),
    ("partial", "run_status", "record_validation_counts", 1,
     [
         "mode",
         "phase",
         "state",
         "artifacts",
         "consolidation_recommended",
         "validation_context",
         "notice",
         "progress",
         "progress_lines",
     ]),
    ("partial", "run_status", "record_high_priority_groups", 1,
     [
         "mode",
         "phase",
         "state",
         "artifacts",
         "consolidation_recommended",
         "validation_context",
         "validation_counts",
         "notice",
         "progress",
         "progress_lines",
     ]),
    ("partial", "run_status", "record_state", 1,
     ["mode", "phase", "progress", "progress_lines"]),
    ("partial-json", "run_status", "_discovery_report", 1,
     []),
    ("partial-json", "status_native", "set_aside_superseded_runs", 1,
     [], CLEAN_LINE),
    ("partial-json", "run_status", "record_mode_phase", 1,
     []),
    ("partial-json", "status_text", "_progress_lines", 1,
     ["mode", "phase", "progress"]),
    ("partial-json", "status_views", "exit_code_for", 1,
     ["mode", "phase", "progress", "progress_lines"]),
    ("partial-json", "status_views", "_result_state", 1,
     ["mode", "phase", "progress", "progress_lines"]),
    ("partial-json", "run_status", "_commit_fixes_view", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial-json", "status_views", "_workflow_payload", 1,
     ["mode", "phase", "state", "progress", "progress_lines"], CLEAN_LINE),
    ("partial-json", "status_native", "state_apply_views", 1,
     ["mode", "phase", "state", "progress", "progress_lines"], CLEAN_LINE),
    ("partial-json", "run_status", "_emit_user_next_step", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial-json", "status_text", "_validation_context_notice", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial-json", "run_status", "record_notice", 1,
     ["mode", "phase", "state", "progress", "progress_lines"]),
    ("partial-json", "run_status", "record_artifact", 1,
     ["mode", "phase", "state", "notice", "progress", "progress_lines"]),
    ("partial-json", "run_status", "record_consolidation_recommended", 1,
     [
         "mode",
         "phase",
         "state",
         "artifacts",
         "notice",
         "progress",
         "progress_lines",
     ]),
    ("partial-json", "run_status", "record_validation_context", 1,
     [
         "mode",
         "phase",
         "state",
         "artifacts",
         "consolidation_recommended",
         "notice",
         "progress",
         "progress_lines",
     ]),
    ("partial-json", "run_status", "record_validation_counts", 1,
     [
         "mode",
         "phase",
         "state",
         "artifacts",
         "consolidation_recommended",
         "validation_context",
         "notice",
         "progress",
         "progress_lines",
     ]),
    ("partial-json", "run_status", "record_high_priority_groups", 1,
     [
         "mode",
         "phase",
         "state",
         "artifacts",
         "consolidation_recommended",
         "validation_context",
         "validation_counts",
         "notice",
         "progress",
         "progress_lines",
     ]),
    ("partial-json", "run_status", "record_state", 1,
     ["mode", "phase", "progress", "progress_lines"]),
    ("listing", "run_status", "_discovery_report", 1,
     []),
    ("listing", "status_native", "set_aside_superseded_runs", 1,
     [], CLEAN_LINE),
    ("listing", "status_views", "exit_code_for", 1,
     []),
    ("listing", "run_status", "_commit_fixes_view", 1,
     []),
    ("listing", "status_views", "_workflow_payload", 1,
     []),
    ("listing", "status_native", "state_implement_view", 1,
     [], CLEAN_LINE),
    ("listing", "status_native", "state_apply_views", 1,
     [], CLEAN_LINE),
    ("listing", "status_text", "render_workflow_text", 1,
     []),
    ("listing", "status_commands", "user_speller", 1,
     []),
    ("listing", "status_text", "render_run_text", 1,
     []),
    ("listing", "status_commands", "render_runner_text", 1,
     []),
    ("listing", "run_status", "record_artifact", 1,
     []),
    ("listing", "detach", "status_command_for", 1,
     ["artifacts"]),
    ("listing", "run_status", "record_next", 1,
     ["artifacts"]),
    ("listing", "run_status", "record_state", 1,
     ["artifacts", "next"]),
    ("listing", "run_status", "keep_state_word", 1,
     ["state", "artifacts", "next"]),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def world(tmp_path, monkeypatch):
    """A plan whose runs' processes all read as alive, no other host."""
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", "MULTI_LLM_HOST_SESSION_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(process_control, "probe", lambda *a, **k: process_control.ALIVE)
    monkeypatch.setattr(process_control, "process_alive", lambda *a, **k: True)
    return tmp_path


def _poll(world: Path, scenario: str, capsys) -> tuple:
    build, argv = SCENARIOS[scenario]
    plan = make_plan(world)
    build(plan)
    capsys.readouterr()
    code = multi_llm.main(["status", str(plan), "--host", "claude-code", *argv])
    return code, result_payload(capsys.readouterr().out, world)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_the_clean_poll(world, capsys, scenario):
    code, payload = _poll(world, scenario, capsys)
    assert (code, payload) == (CLEAN[scenario]["exit"], CLEAN[scenario])


def _cases():
    for scenario, owner, name, nth, kept, *on_oserror in FAULTS:
        for exc, holds in ((OSError, on_oserror[0] if on_oserror else kept),
                           (KeyboardInterrupt, kept)):
            yield pytest.param(
                scenario, owner, name, nth, exc, holds,
                id=f"{scenario}-{name}-{nth}-{exc.__name__}")


@pytest.mark.parametrize("scenario, owner, name, nth, exc, kept", list(_cases()))
def test_a_fault_keeps_what_was_recorded_before_it(
        world, capsys, monkeypatch, scenario, owner, name, nth, exc, kept):
    module = OWNERS[owner]
    real = getattr(module, name)
    calls = []

    def fault(*args, **kwargs):
        calls.append(name)
        if len(calls) == nth:
            raise exc("injected")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, fault)
    code, payload = _poll(world, scenario, capsys)
    assert len(calls) >= nth, f"{owner}.{name} was called {len(calls)} time(s)"
    if kept == CLEAN_LINE:
        assert (code, payload) == (CLEAN[scenario]["exit"], CLEAN[scenario])
    else:
        assert (code, payload) == (1, expected(scenario, kept))


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX SIGINT")
def test_a_real_sigint_inside_render_run_text(world, capsys, monkeypatch):
    """The interrupt a terminal's Ctrl-C delivers, not a raised one."""
    def interrupted(*args, **kwargs):
        signal.raise_signal(signal.SIGINT)
        raise AssertionError("SIGINT did not interrupt the render")

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        monkeypatch.setattr(status_text, "render_run_text", interrupted)
        code, payload = _poll(world, "failed", capsys)
    finally:
        signal.signal(signal.SIGINT, previous)
    kept = next(row[4] for row in FAULTS if row[:3] == ("failed", "status_text", "render_run_text"))
    assert (code, payload) == (1, expected("failed", kept))
