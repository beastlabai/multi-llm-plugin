"""Dispatch counts in the check-gate world: the ``[ETA]`` lines, the native emit and the util calls.

``_run(repo, capsys, *flags)`` runs ``implement`` and returns its exit code and
output, ``_emit`` the native emit, ``_util`` a ``multi_llm.py util`` command,
and ``_etas`` reads every ``[ETA] implement: task N/M`` line. No fixtures.
"""

from __future__ import annotations

import json
import re

import multi_llm

from tests.support.gate_fixture import PLAN as CHECK_GATE_PLAN
from tests.support.implement_world import _implement


#: Implement's `[ETA]` lines count tasks and say so (fix round 22).
ETA = re.compile(r"\[ETA\] implement: task (\d+)/(\d+) done")


def _etas(text: str) -> list:
    """Every ``[ETA] implement: task N/M`` line's ``(N, M)``, in order."""
    return [(int(a), int(b)) for a, b in ETA.findall(text)]


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _emit(repo, capsys, *flags):
    import implement_orchestrator

    assert implement_orchestrator.main(
        ["--plan-file", str(repo.plan), "--yes", *flags]
    ) == 0
    out = capsys.readouterr().out
    path = next(
        line.split(": ", 1)[1] for line in out.splitlines()
        if line.startswith("Output file: ")
    )
    return json.loads(open(path, encoding="utf-8").read()), out


#: The live fixture's layout (run 8): T007 writes its own file, so it sits in
#: batch 2 beside T004, before the refused T005, and the run's dispatch walk
#: counts it as waiting on T002's verdict.
LIVE_LAYOUT = CHECK_GATE_PLAN.replace(
    "Write the note.\n- Depends on: T002\n- Files to modify: src/main.py",
    "Write the note.\n- Depends on: T002\n- Files to modify: docs/release.md",
)


def _run(repo, capsys, *flags) -> tuple:
    code = _implement(repo, *flags)
    captured = capsys.readouterr()
    return code, captured.out + captured.err
