#!/usr/bin/env python3
"""``util write-prompt``: one write batch's prompt, bound and ready to dispatch.

The native leg of the four write phases (apply-suggestions,
apply-task-suggestions, apply-code-fixes, implement) hands its batches to the
host, which spawns one subagent per batch. The batch prompts live inside
``orchestrator_output.json`` / ``implementation_tasks_*.json`` and carry a
literal ``{prior_changes_context}`` token the host had to bind itself, from a
``util prior-changes read`` it ran first. Measured on the 2026-10-04 Claude
Code racing re-run: every Task subagent spent three to six tool calls guessing
the JSON shape (``tasks``, ``ts``, ``batches[].tasks[]``) to find its prompt,
and in two phases a subagent skipped the prior changes or read ``tasks.md``
instead of the prompt. Review validation already had ready ``*.prompt.txt``
files; the write phases had none.

This command is that file for a write batch. The orchestrator materializes
``{phase}/write_tasks.json`` on the native leg too (the executor's own task
file, through the same ``utils/write_phase_bridge``), and for one task id
this binds the prompt **late** — against the prior-changes log as it is on
disk at that moment, so it reflects exactly the batches already finished —
and appends the write-batch result contract. Both invokers therefore produce
byte-identical text for the same state: it calls the scheduler's own
:func:`utils.write_batch_scheduler.bind_prior_changes` and
:func:`utils.write_batch_scheduler.with_result_contract`, and
``tests/test_write_prompt.py`` pins the two renderings against each other.

Usage::

    multi_llm.py util write-prompt --phase-dir <dir> --task-id <id> [--output PATH]
    multi_llm.py util write-prompt --phase-dir <dir> --task-id <id> --stdout

By default the prompt is written to ``{phase_dir}/prompts/{task_id}.prompt.txt``
(over-written on every call: a ``--resume`` rebinds it) and three marker lines
go to stdout — ``[WRITE_PROMPT] <path>``, ``[WRITE_PROMPT_LINES] <n>`` and
``[WRITE_PROMPT_CONTEXT] <entries> prior-batch entries bound`` — so the host
can hand the subagent "read this file in full and follow it", the contract
the review phases' ``prompt_path`` already uses. ``--stdout`` prints the
prompt text itself instead, and nothing else. Exit 3 names the problem: no
``write_tasks.json`` (run the phase's orchestrator first), an unknown task id
(the known ids are listed), a human task (it never dispatches) or a prompt
without the token (a producer bug, the scheduler's own refusal).

The prior-changes log is the plan dir's ``prior_changes.jsonl`` — the parent
of the phase dir, where the executor leg appends — scoped to this phase as
the scheduler scopes it; the instructions' ``util prior-changes append`` step
names the same directory.

Import is side-effect free: argparse and all I/O live inside :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import prior_changes as _prior_changes
    from .stream_bootstrap import bootstrap_streams
    from .subagent_executor import ExecutorError, ResultRejected
    from .subagent_tasks import PHASE_BY_WRITE_KIND, TASKS_SCHEMA_VERSION
    from .write_batch_scheduler import bind_prior_changes, with_result_contract
    from .write_phase_bridge import WRITE_TASK_FILENAME
except ImportError:  # pragma: no cover - direct script invocation
    from utils import prior_changes as _prior_changes  # type: ignore
    from utils.stream_bootstrap import bootstrap_streams  # type: ignore
    from utils.subagent_executor import ExecutorError, ResultRejected  # type: ignore
    from utils.subagent_tasks import PHASE_BY_WRITE_KIND, TASKS_SCHEMA_VERSION  # type: ignore
    from utils.write_batch_scheduler import (  # type: ignore
        bind_prior_changes,
        with_result_contract,
    )
    from utils.write_phase_bridge import WRITE_TASK_FILENAME  # type: ignore


__all__ = [
    "PROMPTS_SUBDIR",
    "WritePromptError",
    "load_write_task",
    "render_write_prompt",
    "default_output_path",
    "main",
]

#: Where the bound prompt files land, under the phase dir.
PROMPTS_SUBDIR = "prompts"

#: The marker lines a host reads (``--stdout`` prints none of them).
MARKER_PATH = "[WRITE_PROMPT]"
MARKER_LINES = "[WRITE_PROMPT_LINES]"
MARKER_CONTEXT = "[WRITE_PROMPT_CONTEXT]"


class WritePromptError(Exception):
    """A refusal: exit 3 ("needs a human", D6) with one message."""

    exit_code = 3


def _read_document(phase_dir: Path) -> Dict[str, Any]:
    path = phase_dir / WRITE_TASK_FILENAME
    if not path.is_file():
        raise WritePromptError(
            f"No {WRITE_TASK_FILENAME} under {phase_dir}: the phase's "
            f"orchestrator writes it when it emits its batches, so run the "
            f"phase (or re-run it with --resume) before asking for a batch "
            f"prompt."
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WritePromptError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise WritePromptError(f"{path} is not a JSON object.")
    version = document.get("tasks_schema_version")
    if version != TASKS_SCHEMA_VERSION:
        raise WritePromptError(
            f"{path} carries tasks_schema_version {version!r}, not "
            f"{TASKS_SCHEMA_VERSION}: re-run the phase's orchestrator to "
            f"re-materialize it."
        )
    return document


def load_write_task(phase_dir: Path, task_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(document, task)`` for *task_id* from the phase dir's task file."""
    document = _read_document(phase_dir)
    tasks = document.get("batches") or document.get("tasks") or []
    known: List[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("task_id") or "")
        known.append(tid)
        if tid == task_id:
            return document, task
    raise WritePromptError(
        f"No task {task_id!r} in {phase_dir / WRITE_TASK_FILENAME}. Known "
        f"task ids, in dispatch order: {', '.join(known) or '(none)'}."
    )


def _phase_word(document: Dict[str, Any]) -> Optional[str]:
    phase = document.get("phase")
    if isinstance(phase, str) and phase:
        return phase
    kind = document.get("kind")
    return PHASE_BY_WRITE_KIND.get(str(kind or ""))


def render_write_prompt(
    phase_dir: Path, task_id: str
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """The bound prompt text plus ``(task, info)``.

    ``info`` carries ``kind``, ``phase``, ``output_dir`` (the log's dir) and
    ``context_entries`` (how many of this phase's prior-batch entries the
    render could see). The prompt is exactly
    ``with_result_contract(kind, task, bind_prior_changes(task, …))`` — the
    executor leg's dispatch text for the same task and the same log.
    """
    phase_dir = Path(os.path.abspath(str(phase_dir)))
    document, task = load_write_task(phase_dir, task_id)
    kind = str(document.get("kind") or "")
    if kind not in PHASE_BY_WRITE_KIND:
        raise WritePromptError(
            f"{phase_dir / WRITE_TASK_FILENAME} is a {kind!r} task file, not a "
            f"write phase's (known: {', '.join(PHASE_BY_WRITE_KIND)})."
        )
    if task.get("is_human") or str(task.get("executor_role") or "") == "human":
        raise WritePromptError(
            f"Task {task_id} is a human task: it never dispatches to a "
            f"subagent, so there is no prompt to bind. Record its answer "
            f"through the phase's human-task strategy instead."
        )
    phase = _phase_word(document)
    # The log lives in the plan dir — the phase dir's parent, where the
    # executor leg's `WriteBatchScheduler` (output_dir=<plan dir>) appends.
    output_dir = str(phase_dir.parent)
    try:
        bound = bind_prior_changes(task, output_dir=output_dir, phase=phase)
    except ResultRejected as exc:
        raise WritePromptError(str(getattr(exc, "message", exc))) from exc
    text = with_result_contract(kind, task, bound)
    scoped = _prior_changes.canonical_phase(phase)
    entries = [
        entry for entry in _prior_changes.entries(output_dir)
        if _prior_changes.canonical_phase(entry.get("phase")) == scoped
    ]
    info = {
        "kind": kind,
        "phase": phase,
        "output_dir": output_dir,
        "context_entries": len(entries),
    }
    return text, task, info


def default_output_path(phase_dir: Path, task_id: str) -> Path:
    """``{phase_dir}/prompts/{task_id}.prompt.txt``."""
    return Path(os.path.abspath(str(phase_dir))) / PROMPTS_SUBDIR / f"{task_id}.prompt.txt"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi_llm.py util write-prompt",
        description=(
            "Print or write one write batch's prompt, with the prior-batch "
            "context bound from the log as it is now and the write-batch "
            "result contract appended — the text the executor leg would "
            "dispatch for the same task."
        ),
    )
    parser.add_argument(
        "--phase-dir", required=True,
        help="The phase directory holding write_tasks.json (e.g. "
             "plans/my-plan/apply-suggestions, plans/my-plan/implement).",
    )
    parser.add_argument(
        "--task-id", required=True,
        help="The task id from write_tasks.json (apply-batch-0, "
             "apply-task-batch-0, apply-fix-batch-0, T001, ...).",
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "--output", default=None, metavar="PATH",
        help=f"Write the prompt here instead of {PROMPTS_SUBDIR}/<task-id>.prompt.txt "
             f"under the phase dir.",
    )
    where.add_argument(
        "--stdout", action="store_true",
        help="Print the prompt text itself and nothing else.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    bootstrap_streams()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        text, _task, info = render_write_prompt(Path(args.phase_dir), args.task_id)
    except (WritePromptError, ExecutorError) as exc:
        message = getattr(exc, "message", None) or str(exc)
        print(f"ERROR: {message}", file=sys.stderr)
        return int(getattr(exc, "exit_code", 3) or 3)

    if args.stdout:
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0

    target = Path(args.output) if args.output else default_output_path(
        Path(args.phase_dir), args.task_id
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot write {target}: {exc}", file=sys.stderr)
        return 1
    lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    print(f"{MARKER_PATH} {target}")
    print(f"{MARKER_LINES} {lines}")
    print(
        f"{MARKER_CONTEXT} {info['context_entries']} prior-batch entries bound "
        f"({info['phase']}, from {Path(info['output_dir']) / _prior_changes.FILENAME})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
