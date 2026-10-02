#!/usr/bin/env python3
"""``util check-tasks``: is a ``tasks.md`` still a consistent task graph?

apply-task-suggestions edits ``tasks.md`` batch by batch, each batch a subagent
that reads the file, applies its suggestions and regenerates the dependency
graph by hand. The 2026-10-01 Claude Code run ended with T005 now defining
``projectProgress`` while T006 still said to implement it, and a new T012
appended last although T001 depended on it (three-host racing run, issue 10).
Nothing re-checked the graph after the last batch.

This is that check, and only the part of it a parser can decide: the same graph
rules a generated task list must satisfy
(:func:`utils.task_decomposer.validate_generated_tasks`: unique ``T###`` ids,
every ``depends_on`` declared, acyclic, something ready to start) plus one the
edit batches break, a task that depends on one listed *after* it. It cannot see
two tasks that claim the same work; the batch prompts carry that rule.

Advisory: it prints one ``[TASKS_GRAPH_WARNING]`` line per finding and exits 0
unless the file cannot be read (2), so a finding never fails a phase that did
write its edits.

A task's dependencies are read from its ``**Dependencies**:`` line alone (what
``update_plan_tasks.generate_tasks_markdown`` renders for every task), never from
its prose: ``TaskDecomposer._extract_dependencies`` also matches "after",
"requires" and "depends on" anywhere in the section, so "ticks after 3 frames"
became a dependency on ``T3`` and the check warned about an id nobody wrote
(adversarial review, 2026-10-02).
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional, Sequence

try:
    from .stream_bootstrap import bootstrap_streams
    from .task_decomposer import TaskDecomposer, validate_generated_tasks
except ImportError:  # pragma: no cover - direct-script import shim
    from stream_bootstrap import bootstrap_streams
    from task_decomposer import TaskDecomposer, validate_generated_tasks

WARNING_PREFIX = "[TASKS_GRAPH_WARNING]"

#: A task's metadata line naming its dependencies: ``**Dependencies**: T001, T002``
#: as rendered, plus the ``**Depends on**:`` spelling an edit batch may write.
#: Anchored to a whole line and to the bold label, so a sentence never matches.
_DEPENDENCIES_LINE = re.compile(
    r"^\s*(?:[-*]\s+)?\*{2}\s*(?:dependencies|depends\s+on)\s*"
    r"(?::\s*\*{2}|\*{2}\s*:)(?P<ids>[^\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)


class _RenderedDependencies(TaskDecomposer):
    """The task decomposer with its dependency reading narrowed to the line.

    Everything else (sections, ids, their order) is the decomposer's own, so the
    ids checked are the ids ``implement`` parses.
    """

    def _extract_dependencies(self, description: str) -> List[str]:
        deps: List[str] = []
        for line in _DEPENDENCIES_LINE.finditer(description):
            for token in re.findall(r"\bT?\d+\b", line.group("ids")):
                normalized = token if token.startswith("T") else f"T{token}"
                if normalized not in deps:
                    deps.append(normalized)
        return deps


def tasks_graph_warnings(tasks_markdown: str) -> List[str]:
    """Every inconsistency in a rendered ``tasks.md``'s task graph; empty is fine."""
    tasks = _RenderedDependencies().parse_plan(tasks_markdown)
    if not tasks:
        return []
    records = [{"id": task.id, "depends_on": list(task.depends_on)} for task in tasks]
    warnings = list(validate_generated_tasks(records))
    position = {task.id: index for index, task in enumerate(tasks)}
    for task in tasks:
        later = [dep for dep in task.depends_on if position.get(dep, -1) > position[task.id]]
        if later:
            warnings.append(
                f"{task.id} depends on {', '.join(later)}, listed after it: a task "
                "added by an edit batch belongs where its dependents can follow it"
            )
    return warnings


def main(argv: Optional[Sequence[str]] = None) -> int:
    bootstrap_streams()
    parser = argparse.ArgumentParser(
        prog="multi_llm.py util check-tasks",
        description="Report inconsistencies in a tasks.md's task graph (advisory).",
    )
    parser.add_argument("tasks_file", help="path to tasks.md")
    args = parser.parse_args(argv)
    try:
        with open(args.tasks_file, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        print(f"ERROR: cannot read {args.tasks_file}: {exc}", file=sys.stderr)
        return 2
    warnings = tasks_graph_warnings(text)
    for warning in warnings:
        print(f"{WARNING_PREFIX} {warning}")
    if not warnings:
        print("[TASKS_GRAPH_OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
