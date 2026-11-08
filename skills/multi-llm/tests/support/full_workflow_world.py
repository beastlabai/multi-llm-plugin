"""The full workflow orchestrator over stubbed phases.

``_plan(tmp_path)`` writes a plan with ``PLAN_BODY``.

Fixtures (registered suite-wide by ``tests/conftest.py``): ``stub_phases``, which
stubs every mode-dispatched phase and returns the call log (see its docstring
for ``configure``).
"""

from pathlib import Path

import pytest

import full_workflow_orchestrator as fwo
import modes as mode_table
from utils.cli_common import record_next
from utils.state_manager import StateManager


PLAN_BODY = """# Full Workflow Plan

## Overview
Add a foundation module and wire it into the entry point.

## Step 1
Create `src/foundation.py`.

## Step 2
Call it from `src/main.py`.
"""


def _plan(tmp_path: Path, name: str = "wf-plan") -> Path:
    plan = tmp_path / f"{name}.md"
    plan.write_text(PLAN_BODY, encoding="utf-8")
    return plan


@pytest.fixture
def stub_phases(monkeypatch):
    """Stub every mode-dispatched phase; return the call log.

    ``configure(fail={...}, silent={...})`` shapes the stub:
    ``fail`` maps a mode word to the exit code it returns (nothing recorded), and
    ``silent`` names modes that return 0 *without* recording a completion — the
    native-invoker case a real fan-out produces when it stops at its own pending
    marker.
    """
    calls = []
    state = {"fail": {}, "silent": set()}

    def run_mode(self, mode_word, extra):
        calls.append((mode_word, list(extra)))
        if mode_word in state["fail"]:
            return state["fail"][mode_word]
        if mode_word in state["silent"]:
            record_next(f"<follow-up for {mode_word}>")
            return 0
        phase = mode_table.phase_for(mode_word) or mode_word
        manager = StateManager(self.plan_path)
        manager.mark_phase_completed(phase)
        manager.save()
        return 0

    monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)

    def configure(fail=None, silent=None):
        state["fail"] = dict(fail or {})
        state["silent"] = set(silent or ())
        return calls

    configure()
    return configure
