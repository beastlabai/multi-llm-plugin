"""Next-command emitters: workflow states built from inside the repository, and stubbed mode targets.

``build(workdir, spec)`` is ``tests.support.next_phase_world._build`` run with
the cwd in the skill's repository (a plan's first ``StateManager`` records
``git rev-parse HEAD``); ``_terminal_record`` writes a finished detached run's
checkpoint; ``_target`` stubs every mode's target. ``PLUGIN`` is the Claude Code
plugin route's prefix. No fixtures.
"""

import os
from pathlib import Path

import multi_llm
from utils import status_file
from utils.output_handler import output_dir_path

from tests.support.next_phase_world import _build


SKILL_ROOT = Path(__file__).parent.parent.parent


PLUGIN = "/multi-llm:multi-llm "


def build(workdir, spec):
    """``tests/test_next_phase._build``, from inside the repository.

    A plan's first ``StateManager`` records ``git rev-parse HEAD`` from the
    process's cwd, which the ``workdir`` fixture moved out of any repository.
    """
    here = os.getcwd()
    os.chdir(SKILL_ROOT)
    try:
        return _build(workdir, spec)
    finally:
        os.chdir(here)


def _target(monkeypatch, body=None, code=0):
    """Stub every mode's target: run *body(argv)* (if any), return *code*."""
    seen = {}

    def loader(module_path, attr):
        def target(argv):
            seen["argv"] = list(argv)
            if body is not None:
                result = body(argv)
                if result is not None:
                    return result
            return code

        return target

    monkeypatch.setattr(multi_llm, "_load_target", loader)
    return seen


def _terminal_record(plan, phase_dir_name, **run):
    """A finished detached run's checkpoint, without spawning anything."""
    phase_dir = output_dir_path(plan) / phase_dir_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    status_file.write_status(str(phase_dir), {
        "phase": phase_dir_name, "state": "complete",
    })
    record = {"run_id": "run-t5-0001", "state": "done", "exit_code": 0,
              "detached": True}
    record.update(run)
    status_file.claim_run(str(phase_dir), record)
    return phase_dir
