"""Manual tasks in the write orchestrators' world.

``_manual(repo, *task_ids)`` makes those tasks of ``orchestrator_repo``'s chain
(T001 → T002 → T003) manual. No fixtures.
"""

from __future__ import annotations


def _manual(repo, *task_ids):
    """The repo fixture's chain T001 → T002 → T003 with *task_ids* manual."""
    import implement_orchestrator

    original = implement_orchestrator.TaskDecomposer

    class SeededDecomposer(original):
        def parse_plan(self, content):
            tasks = super().parse_plan(content)
            for task_id in task_ids:
                self.tasks[task_id].executor_role = "human"
                self.tasks[task_id].subagent_type = "human"
            return tasks

    repo.monkeypatch.setattr(implement_orchestrator, "TaskDecomposer", SeededDecomposer)
