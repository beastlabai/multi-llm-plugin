"""A fan-out where every reviewer failed records its failure map.

Plan review-run-scope-salvage-and-failure-visibility, D5.2 / decision 11.11: a
``--reaggregate`` names each remaining failure from ``.status.json``'s
``models_failed`` and exits 3 when every requested reviewer failed. Both
fan-outs used to ``sys.exit`` on the zero-success branch *before* the status
write that records ``models_failed``, so a later reaggregate of that very dir
found no recorded failure: it named the model only with the "no result file"
gap reason and exited 0.

``test:model`` resolves to no provider, so every slot fails without spawning
anything (the same seam ``test_duplicate_run_prevention`` uses).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

from utils.output_handler import get_phase_dir  # noqa: E402

SPEC = "test:model"


def _run(script, plan, *extra):
    return subprocess.run(
        [sys.executable, str(SKILL_DIR / script), "--plan-file", str(plan), *extra],
        capture_output=True, text=True, cwd=str(SKILL_DIR), timeout=120,
        encoding="utf-8",
    )


@pytest.mark.parametrize("script, phase", [
    ("review_plan_orchestrator.py", "review-plan"),
    ("code_review_orchestrator.py", "code-review"),
])
def test_the_failure_map_survives_the_exit_and_the_reaggregate_exits_three(
    tmp_path, script, phase
):
    plan = tmp_path / "test-plan.md"
    plan.write_text("# Test Plan\n\n## Tasks\n1. Task A\n", encoding="utf-8")
    (tmp_path / "test-plan").mkdir()

    fanout = _run(script, plan, "--models", SPEC, "--force")
    assert fanout.returncode == 3, fanout.stdout + fanout.stderr
    assert "ERROR: All models failed" in fanout.stdout

    status = json.loads(
        (get_phase_dir(plan, phase) / ".status.json").read_text(encoding="utf-8")
    )
    assert SPEC in status["models_failed"]
    assert status["models_failed"][SPEC]
    assert status["models_completed"] == []

    reaggregate = _run(script, plan, "--reaggregate")
    assert reaggregate.returncode == 3, reaggregate.stdout + reaggregate.stderr
    assert f"[MODEL_FAILED] {SPEC}: {status['models_failed'][SPEC]}" in reaggregate.stdout
    assert "no result file and no recorded failure" not in reaggregate.stdout
