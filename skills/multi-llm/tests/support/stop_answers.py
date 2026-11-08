"""The report check on a stop's answers, in the check-gate world.

``_native_stop(repo, capsys, check=...)`` runs the native first stop (T005
refused by the emit, T002's check as given); ``_check`` runs the report check
on a transcript of that output and a final message. ``RESUME`` is the answer
command. No fixtures.
"""

from __future__ import annotations

import multi_llm
from utils.state_manager import get_or_create_state

from tests.support.gate_fixture import FINDINGS
from tests.support.transcripts import Transcript, run_check


RESUME = "/multi-llm:multi-llm --implement plans/checked.md --resume"


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _native_stop(repo, capsys, *, check: str) -> str:
    """The native first stop (T005 refused by the emit), T002's check
    recorded *check*; returns `util next-command`'s output."""
    import implement_orchestrator

    assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
    capsys.readouterr()
    phase = str(repo.phase_dir)
    for task_id in ("T001", "T003", "T004"):
        assert _util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                     task_id, "--status", "completed")[0] == 0
    extra = (["--check", "findings", "--finding", FINDINGS[0]]
             if check == "findings" else ["--check", "passed"])
    assert _util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
                 "--status", "completed", *extra)[0] == 0
    state_file = str(get_or_create_state(repo.plan).state_file)
    assert _util(capsys, "finalize-tracking", "--state-file", state_file)[0] == 0
    code, out, err = _util(capsys, "next-command", "plans/checked.md",
                           "--after", "implement")
    assert code == 0, err
    return out


def _check(tmp_path, capsys, output: str, message: str) -> str:
    path = (Transcript().prompt("implement it").bash(output).say(message)
            .write(tmp_path / "t.jsonl"))
    return run_check(capsys, path)
