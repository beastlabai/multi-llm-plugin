"""generate-tasks driven to its pending marker, and the task document every leg accepts.

``_plan(tmp_path)`` writes a plan with ``PLAN_BODY``; ``_materialize(plan)``
runs the phase up to and including the pending marker and returns the task;
``_valid_document()`` is a task list every leg of the loop accepts. No fixtures.
"""

import json
from pathlib import Path

import generate_tasks_orchestrator as gto
import multi_llm


PLAN_BODY = """# Task Generation Plan

## Overview
Add a foundation module and wire it into the entry point.

## Step 1
Create `src/foundation.py`.

## Step 2
Call it from `src/main.py`.
"""


RESULT_NAME = gto.RESULT_NAME
TASK_FILE_NAME = gto.TASK_FILE_NAME


def _plan(tmp_path: Path, name: str = "tg-plan") -> Path:
    plan = tmp_path / f"{name}.md"
    plan.write_text(PLAN_BODY, encoding="utf-8")
    return plan


def _phase_dir(plan: Path) -> Path:
    return plan.parent / plan.stem / "generate-tasks"


def _tasks_md(plan: Path) -> Path:
    return plan.parent / plan.stem / "tasks" / "tasks.md"


def _valid_document() -> dict:
    """The document every leg of the loop is expected to accept."""
    return {
        "schema_version": "1.0",
        "plan_preamble": "Foundation first, then the wiring that consumes it.",
        "tasks": [
            {
                "id": "T001",
                "title": "Create the foundation module",
                "description": "Add src/foundation.py exporting the interface.",
                "depends_on": [],
                "files_to_create": ["src/foundation.py"],
                "files_to_modify": [],
                "acceptance_criteria": ["src/foundation.py exports the interface"],
                "estimated_complexity": "low",
                "executor_role": "agent",
                "subagent_type": None,
            },
            {
                "id": "T002",
                "title": "Wire it into the entry point",
                "description": "Call T001's interface from src/main.py.",
                "depends_on": ["T001"],
                "files_to_create": [],
                "files_to_modify": ["src/main.py"],
                "acceptance_criteria": ["src/main.py calls the interface"],
                "estimated_complexity": "medium",
                "executor_role": "agent",
                "subagent_type": None,
            },
        ],
    }


def _materialize(plan: Path) -> dict:
    """Run the phase up to (and including) the pending marker; return the task."""
    assert multi_llm.main(["generate-tasks", str(plan)]) == 0
    document = json.loads((_phase_dir(plan) / TASK_FILE_NAME).read_text(encoding="utf-8"))
    return document["tasks"][0]
