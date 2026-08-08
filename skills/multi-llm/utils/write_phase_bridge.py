#!/usr/bin/env python3
"""Turn each write phase's existing batch output into materialized write tasks.

The four write phases already build a **complete prompt** per batch in Python
(Pattern A: ``utils/suggestion_batcher.format_batch_for_prompt``,
``utils/code_fix_batcher``, ``apply_task_suggestions_orchestrator``,
``implement_orchestrator.build_task_prompt``). What they *don't* build is the
materialized-task envelope the executor is the second invoker of — the
``tasks_schema_version: 2`` shape with ``output_name``, ``inputs`` hashes,
``executor_role``, and the four write fields D3f/D3g/D3h need.

This module is that one conversion, in one place, so:

* the same declared file set drives the disjointness admission, the write policy
  and the snapshot for every phase (rather than each orchestrator deciding);
* ``prior_changes_id`` is derived from a **stable content key** (section key, file
  key, task id) and never from the loop index, which is what makes the
  prior-changes append idempotent across a re-dispatch;
* adding a write phase is a row in :data:`_ADAPTERS`, not a fifth hand-rolled
  materializer.

Nothing here dispatches: ``utils/write_batch_scheduler`` is the consumer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:  # pragma: no cover - import shim mirrors the rest of utils/
    from .subagent_tasks import (
        PHASE_BY_WRITE_KIND,
        build_inputs,
        make_task_file,
        make_write_task,
        write_task_file,
    )
except ImportError:  # pragma: no cover - direct-script import shim
    from utils.subagent_tasks import (
        PHASE_BY_WRITE_KIND,
        build_inputs,
        make_task_file,
        make_write_task,
        write_task_file,
    )


__all__ = [
    "WRITE_TASK_FILENAME",
    "materialize_write_tasks",
    "write_task_file_for_phase",
]


#: The materialized write task file, beside the phase's own orchestrator output.
#: Named for the phase's kind so a phase dir shared by a resumed run cannot mix two.
WRITE_TASK_FILENAME = "write_tasks.json"


def _stable_key(candidates: Sequence[Any], fallback: str) -> str:
    """The first non-empty candidate, else ``fallback``. Never the loop index."""
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return fallback


def _apply_suggestions_task(
    batch: Dict[str, Any], index: int, context: Dict[str, Any]
) -> Dict[str, Any]:
    """apply-suggestions: one shared artifact (the plan), sequential by invariant."""
    suggestions = batch.get("suggestions") or []
    section = _stable_key([batch.get("section_key")], f"batch-{index}")
    return dict(
        task_id=f"apply-batch-{index}",
        prompt=str(batch.get("prompt") or ""),
        batch_index=index,
        # The plan file is THE artifact: it is what gets snapshotted and what the
        # byte comparison decides on.
        declared_files=[context["plan_file"]],
        expected_ids=[
            _stable_key([s.get("group_id"), s.get("id"), s.get("title")], f"s{n}")
            for n, s in enumerate(suggestions)
        ],
        prior_changes_id=section,
        can_parallelize=False,
        task_title=f"apply {len(suggestions)} suggestion(s) to {section}",
        extra={"section_key": section},
    )


def _apply_task_suggestions_task(
    batch: Dict[str, Any], index: int, context: Dict[str, Any]
) -> Dict[str, Any]:
    """apply-task-suggestions: same shape, but the artifact is the tasks file."""
    suggestions = batch.get("suggestions") or []
    section = _stable_key([batch.get("section_key")], f"batch-{index}")
    artifact = context.get("tasks_file") or context["plan_file"]
    return dict(
        task_id=f"apply-task-batch-{index}",
        prompt=str(batch.get("prompt") or ""),
        batch_index=index,
        declared_files=[artifact],
        expected_ids=[
            _stable_key([s.get("group_id"), s.get("id"), s.get("title")], f"s{n}")
            for n, s in enumerate(suggestions)
        ],
        prior_changes_id=section,
        can_parallelize=False,
        task_title=f"apply {len(suggestions)} task suggestion(s) to {section}",
        extra={"section_key": section},
    )


def _apply_code_fixes_task(
    batch: Dict[str, Any], index: int, context: Dict[str, Any]
) -> Dict[str, Any]:
    """apply-code-fixes: partitioned by ``file_key``, so batches CAN parallelize.

    ``can_parallelize`` is set from the partitioning the batcher already performs
    (one file per batch), but it is only a *permission*: the scheduler still needs
    ``driver.max_concurrent_write_subagents > 1`` and the dispatch-time
    pairwise-disjoint check to agree before two of them run together.
    """
    fixes = batch.get("fixes") or []
    file_key = _stable_key([batch.get("file_key")], f"batch-{index}")
    return dict(
        task_id=f"apply-fix-batch-{index}",
        prompt=str(batch.get("prompt") or ""),
        batch_index=index,
        declared_files=[file_key],
        expected_ids=[
            _stable_key([f.get("id"), f.get("group_id"), f.get("title")], f"f{n}")
            for n, f in enumerate(fixes)
        ],
        prior_changes_id=file_key,
        can_parallelize=bool(batch.get("file_key")),
        task_title=f"apply {len(fixes)} fix(es) to {file_key}",
        executor_role=str(batch.get("executor_role") or "agent"),
        extra={"file_key": file_key},
    )


def _implement_task(
    task: Dict[str, Any], index: int, context: Dict[str, Any]
) -> Dict[str, Any]:
    """implement: one materialized task per decomposed task, keeping its edges.

    ``depends_on`` travels so the human-task strategies can walk the graph, and
    ``files_to_create``/``files_to_modify`` become the declared set — the same
    partitioning ``implement_orchestrator`` already batches by.
    """
    task_id = _stable_key([task.get("id")], f"task-{index}")
    declared = [
        *(task.get("files_to_modify") or []),
        *(task.get("files_to_create") or []),
    ]
    return dict(
        task_id=task_id,
        prompt=str(task.get("prompt") or ""),
        batch_index=int(context["batch_index"]),
        declared_files=declared,
        expected_ids=[task_id],
        prior_changes_id=task_id,
        can_parallelize=bool(context.get("can_parallelize")),
        depends_on=[str(d) for d in (task.get("depends_on") or [])],
        executor_role=str(task.get("executor_role") or "agent"),
        task_title=str(task.get("title") or task_id),
    )


#: ``kind -> (per-item builder, how the phase's payload is iterated)``.
_ADAPTERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "apply_suggestions": _apply_suggestions_task,
    "apply_task_suggestions": _apply_task_suggestions_task,
    "apply_code_fixes": _apply_code_fixes_task,
    "implement": _implement_task,
}


def materialize_write_tasks(
    *,
    kind: str,
    batches: Sequence[Dict[str, Any]],
    phase_dir: str,
    plan_file: str,
    tasks_file: Optional[str] = None,
    grouped_file: Optional[str] = None,
    output_name_index: Optional[Callable[[int], int]] = None,
) -> List[Dict[str, Any]]:
    """Convert a phase's batch payload into materialized write tasks (D2g shape).

    Args:
        batches: For the three apply kinds, the orchestrator's ``batches`` list
            (each carrying ``prompt``). For ``implement``, the ``batches`` list of
            ``{batch_index, can_parallelize, tasks: [...]}``.

    Returns:
        The task list, validated against ``schemas/subagent_tasks.schema.json``.
    """
    if kind not in _ADAPTERS:
        raise ValueError(f"{kind!r} is not a write kind this bridge materializes")
    inputs = build_inputs(plan_file=plan_file, grouped_file=grouped_file)
    context: Dict[str, Any] = {
        "plan_file": str(Path(plan_file).resolve()) if plan_file else "",
        "tasks_file": str(Path(tasks_file).resolve()) if tasks_file else None,
    }
    builder = _ADAPTERS[kind]
    tasks: List[Dict[str, Any]] = []

    if kind == "implement":
        for batch in batches:
            batch_index = int(batch.get("batch_index", 0))
            for position, inner in enumerate(batch.get("tasks") or []):
                fields = builder(inner, position, {
                    **context,
                    "batch_index": batch_index,
                    "can_parallelize": batch.get("can_parallelize"),
                })
                # Two tasks in one batch cannot share an output_name, so the result
                # file is keyed by a MONOTONIC index across the phase rather than by
                # ``batch_index``, which repeats within a parallelizable batch.
                # Mixing the two spellings is how one task silently overwrites
                # another's result — and D3f's containment would happily allow it,
                # because both names match the kind's pattern.
                tasks.append(make_write_task(
                    kind=kind, inputs=inputs, phase_dir=phase_dir,
                    output_name=f"implement_batch_{len(tasks)}.json",
                    **fields,
                ))
        return tasks

    for index, batch in enumerate(batches):
        fields = builder(batch, index, context)
        tasks.append(make_write_task(
            kind=kind, inputs=inputs, phase_dir=phase_dir, **fields
        ))
    return tasks


def write_task_file_for_phase(
    *,
    kind: str,
    tasks: Sequence[Dict[str, Any]],
    phase_dir: str,
    plan_file: str,
    grouped_file: Optional[str] = None,
) -> str:
    """Persist the phase's materialized write tasks. Returns the path written.

    The file is written even under ``--dry-run-subagents``: it is the input a
    ``--resume`` re-reads and the artifact that makes "what exactly would have been
    dispatched" inspectable without re-deriving it from an orchestrator run.
    """
    document = make_task_file(
        kind=kind,
        tasks=list(tasks),
        inputs=build_inputs(plan_file=plan_file, grouped_file=grouped_file),
        task_list_key="batches",
        extra={
            "total_batches": len(tasks),
            "phase": PHASE_BY_WRITE_KIND[kind],
            "phase_dir": str(phase_dir),
        },
    )
    os.makedirs(phase_dir, exist_ok=True)
    path = os.path.join(phase_dir, WRITE_TASK_FILENAME)
    write_task_file(path, document)
    return path
