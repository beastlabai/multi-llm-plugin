"""Write-task files inside a git repository, for ``util write-prompt``.

Like ``tests.support.batch_done_world``, but ``PROMPT`` has the full apply
layout (prior changes, suggestions, return format), and the plan lives in a
repository, because a shared-artifact batch's snapshot lives under its root.

Fixtures: ``prompt_plan_dir`` provides ``plan_dir`` (``(plan, out)`` in a fresh
repository). It is not registered suite-wide, because
``tests.support.batch_done_world`` owns the suite's ``plan_dir``; import it to
use it.
"""

import subprocess

import pytest

from utils.subagent_tasks import PRIOR_CHANGES_TOKEN
from utils.write_phase_bridge import materialize_write_tasks, write_task_file_for_phase


PROMPT = (
    "# Apply these suggestions\n\n"
    "## Prior changes\n\n" + PRIOR_CHANGES_TOKEN + "\n\n"
    "## Suggestions\n\n1. Add email validation\n2. Enforce password rules\n\n"
    "## Return Format\n\nSummarize each item.\n"
)


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
            "prompt": PROMPT.replace("Add email validation", "Add rate limiting"),
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


@pytest.fixture(name="plan_dir")
def prompt_plan_dir(tmp_path):
    # A repository: a shared-artifact batch's snapshot lives under its root,
    # and outside one `util write-prompt` refuses the batch.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    plan = tmp_path / "plans" / "feature.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Feature\n\n## Step 2\n\nBody.\n", encoding="utf-8")
    out = tmp_path / "plans" / "feature"
    out.mkdir()
    return plan, out


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
