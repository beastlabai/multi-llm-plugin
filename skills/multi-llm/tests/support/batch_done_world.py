"""Write-task files for ``util batch-done``: a plan, its output folder, and materialized batches.

``_materialize(plan, out, kind, batches, phase_subdir)`` writes a phase's
write-task file from ``_suggestion_batches()`` or ``_implement_batches()``
(every prompt carries the prior-changes token); ``_state(out)`` reads the
plan's ``state.json``. ``INSTRUCTIONS`` is the skill's ``instructions/`` folder.

Fixtures (registered suite-wide by ``tests/conftest.py``): ``plan_dir``, a
``(plan, out)`` pair under ``tmp_path`` (no repository; see
``tests.support.write_prompt_world.prompt_plan_dir`` for one inside a repository).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.subagent_tasks import PRIOR_CHANGES_TOKEN
from utils.write_phase_bridge import materialize_write_tasks, write_task_file_for_phase


INSTRUCTIONS = Path(__file__).parent.parent.parent / "instructions"

PROMPT = "# Apply\n\n" + PRIOR_CHANGES_TOKEN + "\n\n1. Do the thing.\n"


@pytest.fixture
def plan_dir(tmp_path):
    plan = tmp_path / "plans" / "feature.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Feature\n\n## Step 2\n\nBody.\n", encoding="utf-8")
    out = tmp_path / "plans" / "feature"
    out.mkdir()
    return plan, out


def _suggestion_batches():
    return [
        {
            "section_key": "step_2",
            "prompt": PROMPT,
            "suggestions": [
                {"group_id": "g-email", "title": "Add email validation"},
                {"group_id": "g-password", "title": "Enforce password rules"},
            ],
        },
        {
            "section_key": "step_3",
            "prompt": PROMPT,
            "suggestions": [{"group_id": "g-rate", "title": "Add rate limiting"}],
        },
    ]


def _implement_batches():
    return [
        {"batch_index": 0, "can_parallelize": False, "tasks": [
            {"id": "T001", "title": "Foundation", "prompt": PROMPT,
             "files_to_create": ["src/a.py"], "depends_on": [],
             "executor_role": "agent"},
        ]},
        {"batch_index": 1, "can_parallelize": False, "tasks": [
            {"id": "T002", "title": "Manual step", "prompt": PROMPT,
             "files_to_create": [], "depends_on": ["T001"],
             "executor_role": "human"},
        ]},
    ]


def _materialize(plan, out, kind, batches, phase_subdir):
    phase_dir = out / phase_subdir
    phase_dir.mkdir(parents=True, exist_ok=True)
    tasks = materialize_write_tasks(
        kind=kind, batches=batches, phase_dir=str(phase_dir), plan_file=str(plan),
    )
    write_task_file_for_phase(
        kind=kind, tasks=tasks, phase_dir=str(phase_dir), plan_file=str(plan),
    )
    return phase_dir, tasks


def _state(out):
    return json.loads((out / "state.json").read_text(encoding="utf-8"))
