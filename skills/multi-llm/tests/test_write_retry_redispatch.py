"""A failed write attempt is never sent again on top of its own edits.

Fix round 8 (review 7, r5): an ``implement`` batch that edited ``src/main.py``
and then replied in prose was recorded ``unparseable_result``, which is on the
transport retry rung, so the executor re-sent the byte-identical batch prompt
and the edit landed twice. The pre-retry changed-files check (T4.8) ran only for
a turn a denial ended. It now runs before every re-dispatch of a write batch's
own prompt — a timeout, a runner error, a prose reply, a rate limit — and an
attempt that moved bytes is never re-sent: it is a partial application (loop
stops, snapshot kept, exit 3), or, for an apply kind whose reply could not be
read, the one report re-ask, which restates the report and edits nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import implement_orchestrator  # noqa: E402
from tests.test_write_executor import _patch_edit, env  # noqa: E402,F401
from utils import provider_registry as registry  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils.output_handler import get_phase_dir  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402
from utils.subagent_tasks import PRIOR_CHANGES_TOKEN  # noqa: E402

MOCK_LLM = Path(__file__).parent / "mocks" / "mock_llm.py"
REASK_MARKER = "You are re-sending the report of a batch of edits."
VERDICT_WARNING = "could not record the verdict"

GOOD = {"status": "applied", "ids": ["s0"], "files_touched": ["plans/p.md"],
        "summary": "added the step"}
PROSE = "I went ahead and made the changes. Everything looks good now!"


def _prompts(log: Path):
    if not log.exists():
        return []
    return [json.loads(line)["prompt"] for line in log.read_text().splitlines()]


# ===========================================================================
# A dispatch that persisted no result file has no verdict to stamp
# ===========================================================================

def test_a_failed_dispatch_with_no_result_file_draws_no_verdict_warning(env):
    """``output_path`` names where the result would go; a prose reply never
    wrote it, and the stamp used to warn ``could not record the verdict on
    apply_batch_0.json … No such file`` about a file nobody meant to write."""
    env.edits([], prose=True)
    lines = []

    result = env.scheduler([env.apply_task(0)], log=lines.append).run()

    assert result.outcomes[0].status == executor.STATUS_UNPARSEABLE
    assert not (env.phase_dir / "apply_batch_0.json").exists()
    assert not [line for line in lines if VERDICT_WARNING in line], lines
