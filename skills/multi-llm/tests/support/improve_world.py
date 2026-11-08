"""The improve composites' world: improve-plan and improve-code over stubbed phases.

``stub`` replaces every phase ``run_mode`` dispatches and returns
``configure(fail=..., silent=..., hook=...)``; each call is logged as
``(mode word, argv, the step lock's holder while it ran)``. ``_plan``,
``_write_state`` and ``_write_manifest`` lay out a plan's state on disk;
``_live_run`` holds a phase lock as a live run would.

Fixtures: ``stub`` (registered suite-wide by ``tests/conftest.py``), and
``improve_repo``, which provides ``repo`` (a git repository at the cwd with one
empty commit). ``improve_repo`` is not registered, because
``tests.support.gate_fixture`` owns the suite's ``repo``; import it to use it.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import full_workflow_orchestrator as fwo
import improve_code_orchestrator as ico
import improve_plan_orchestrator as ipo
import modes as mode_table
from utils import detach, process_control, status_file
from utils import improve_workflow as iw
from utils.output_handler import output_dir_path
from utils.state_manager import StateManager


#: A `--files-from` listing whose name carries both quote kinds. Windows forbids
#: `"` in a filename, so there it keeps only the `'`; the `--files` names,
#: which never touch the disk, still carry both.
LISTING_NAME = "list 'x' \"y\".txt" if os.name != "nt" else "list 'x' y.txt"


PLAN = "improve-plan"
CODE = "improve-code"
MODES = [PLAN, CODE]
ENTRY = {PLAN: ipo, CODE: ico}

#: The review and apply phase word of each composite, and their dirs.
REVIEW_PHASE = {PLAN: "review-plan", CODE: "code-review"}
APPLY_PHASE = {PLAN: "apply-suggestions", CODE: "apply-code-fixes"}


APPLY_FLAGS = ["--yes", "--agent-decide", "--no-confirm"]


BASE = datetime(2026, 10, 1, 12, 0, 0)


def T(minutes: float) -> str:
    """A naive local stamp *minutes* after a fixed base."""
    return (BASE + timedelta(minutes=minutes)).isoformat()


@pytest.fixture(name="repo")
def improve_repo(tmp_path, monkeypatch):
    """A git repository the plan lives in, and the cwd: review-code's scope
    reader resolves ``--files`` against the plan's repository root."""
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _plan(root: Path, name: str = "imp-plan") -> Path:
    plan = root / "plans" / f"{name}.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# Improve\n\n## Step 1\nDo the thing.\n", encoding="utf-8")
    return plan


def _out(plan: Path) -> Path:
    return output_dir_path(plan)


def _write_state(plan: Path, *, completed=None, skipped=None, started=None,
                 extra=None) -> Path:
    """``state.json`` with exactly these records (stamps as given)."""
    state = StateManager(plan)
    state.state["phases_completed"] = dict(completed or {})
    state.state["phases_skipped"] = {
        phase: ({"skipped_at": value, "reason": "test"}
                if isinstance(value, str) or value is None else value)
        for phase, value in (skipped or {}).items()
    }
    if started:
        state.state["phases_started"] = dict(started)
    for key, value in (extra or {}).items():
        state.state[key] = value
    state.save()
    return state.state_file


def _args(mode: str, plan: Path, *argv: str) -> argparse.Namespace:
    return iw.parse_args(mode, ["--plan-file", str(plan), *argv])


def _write_manifest(plan: Path, base_ref: str) -> Path:
    """A ``code-review/review-scope.md`` header ``read_review_scope`` accepts."""
    phase_dir = _out(plan) / "code-review"
    phase_dir.mkdir(parents=True, exist_ok=True)
    path = phase_dir / "review-scope.md"
    path.write_text(
        "# Review scope\n\n"
        "- Scope id: `0123abcd`\n"
        f"- Base ref: `{base_ref}` (commit `abc1234`)\n"
        "- Written by: review-code\n\n"
        "Diff coverage: complete\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(name="stub")
def improve_stub(monkeypatch):
    """Stub every phase ``run_mode`` dispatches; return ``configure``.

    ``configure(fail={mode: code}, silent={mode})`` shapes it as
    ``test_full_workflow_orchestrator``'s ``stub_phases``. Each call is logged
    as ``(mode word, argv, the step lock's holder while it ran)``; a consolidation
    pass (``--consolidate``) records nothing.
    """
    calls = []
    shape = {"fail": {}, "silent": set(), "hook": None}

    def run_mode(self, mode_word, extra):
        phase = mode_table.phase_for(mode_word) or mode_word
        lock = detach.lock_path(str(self.plan_dir / {
            "apply-code-fixes": "apply-fixes"}.get(phase, phase)))
        held = None
        if not process_control.lock_is_free(lock):
            held = process_control.read_lock_holder(lock)
        calls.append((mode_word, list(extra), held))
        if shape["hook"] is not None:
            shape["hook"](self, mode_word, list(extra))
        if mode_word in shape["fail"]:
            return shape["fail"][mode_word]
        if mode_word in shape["silent"] or "--consolidate" in extra:
            return 0
        manager = StateManager(self.plan_path)
        manager.mark_phase_completed(phase)
        manager.save()
        return 0

    monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)

    def configure(fail=None, silent=None, hook=None):
        shape["fail"] = dict(fail or {})
        shape["silent"] = set(silent or ())
        shape["hook"] = hook
        return calls

    configure()
    return configure


def _argv_of(calls, mode_word):
    found = [argv for word, argv, _held in calls if word == mode_word
             and "--consolidate" not in argv]
    assert len(found) == 1, (mode_word, calls)
    return found[0]


def _main(mode: str, plan: Path, *argv: str) -> int:
    return ENTRY[mode].main(["--plan-file", str(plan), *argv])


def _live_run(directory: Path, run_id: str, mode: str) -> process_control.PhaseLock:
    """A held ``.lock`` plus a live ``run`` record in *directory* (this process)."""
    directory.mkdir(parents=True, exist_ok=True)
    status_file.claim_run(str(directory), {
        "run_id": run_id, "pid": os.getpid(),
        "process_start_time": process_control.process_start_time(os.getpid()),
        "heartbeat_at": status_file.now_iso(), "heartbeat_interval_s": 10.0,
        "state": "running", "detached": True, "mode": mode,
    })
    lock = process_control.PhaseLock(detach.lock_path(str(directory)))
    lock.acquire({"run_id": run_id, "pid": os.getpid(),
                  "process_start_time": process_control.process_start_time(os.getpid())})
    return lock


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


REFUSAL = {
    PLAN: ("ERROR: improve-plan runs unattended and does not take --interactive "
           "(-i). Run --review-plan --interactive, then --apply-suggestions."),
    CODE: ("ERROR: improve-code runs unattended and does not take --interactive "
           "(-i). Run --review-code --interactive, then --apply-code-fixes."),
}


STATE_FAULTS = {
    "not-json": ("{\"phases_completed\": {,}", "Expecting property name"),
    "top-level-list": ("[1, 2]", "the top-level value is a list, not an object"),
    "phases-completed-list": ('{"phases_completed": ["review-plan"]}',
                              "phases_completed is a list, not an object"),
    # Fable review 2026-10-09: every other object the records read is checked
    # too. A `task_status` list passed and then raised in
    # `apply_phase_in_progress` (`.items()`), a traceback at exit 1.
    "task-status-list": ('{"phases_completed": {}, "task_status": ["T001"]}',
                         "task_status is a list, not an object"),
    "task-status-null": ('{"task_status": null}', "task_status is null, not an object"),
    "phases-started-string": ('{"phases_started": "review-plan"}',
                              "phases_started is a string, not an object"),
    "metrics-progress-list": ('{"metrics_progress": []}',
                              "metrics_progress is a list, not an object"),
    "metrics-number": ('{"metrics": 7}', "metrics is a number, not an object"),
}
