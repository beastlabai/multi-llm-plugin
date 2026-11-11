"""`get_workflow_status`'s payload, byte for byte (code health ENTRY-10).

The function is a driver over named passes that run in a fixed order, each
reading what the ones before it wrote. ``golden/workflow_status_cases.json``
holds states recorded from the test suite while it was one function: each
case is a ``state.json``, the review files the passes read, the early warning
the case's test had injected, and the payload that version returned. The
cases cover every pass: recorded and stale entries, each skip, implement's
progress, manual gate and four finding lists, apply phases in progress, the
task-groups notice, the scope blocker and deferral, and the warning.

Compared as JSON text, so the order of an entry's keys (which ``status
--json`` prints) is pinned too. ``<PLAN>`` and ``<PLAN_DIR>`` stand for the
plan and its output directory.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import check_workflow_prerequisites as prereq  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

CASES = json.loads(
    (Path(__file__).parent / "golden" / "workflow_status_cases.json").read_text(encoding="utf-8")
)


def _placed(value, plan: Path, plan_dir: Path):
    text = json.dumps(value)
    text = text.replace("<PLAN_DIR>", json.dumps(str(plan_dir))[1:-1])
    text = text.replace("<PLAN>", json.dumps(str(plan))[1:-1])
    return json.loads(text)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_payload_is_unchanged(name, tmp_path, monkeypatch):
    case = CASES[name]
    plan = tmp_path / "plans" / case["plan_name"]
    plan.parent.mkdir(parents=True)
    plan.write_text("# plan\n", encoding="utf-8")
    plan_dir = output_dir_path(plan)
    for rel, text in _placed(case["files"], plan, plan_dir).items():
        target = plan_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "state.json").write_text(
        json.dumps(_placed(case["state"], plan, plan_dir)), encoding="utf-8")
    asked = []

    def warning(*_args, **_kwargs):
        asked.append(True)
        return case["warning"]

    monkeypatch.setattr(prereq, "repo_wide_sandbox_warning", warning)
    state = StateManager(plan, read_only=True)

    result = prereq.get_workflow_status(state, plan, plan_dir)

    assert json.dumps(result, indent=1) == json.dumps(
        _placed(case["expected"], plan, plan_dir), indent=1)
    # The warning is asked only while a repo-wide write phase is open.
    assert bool(asked) == case["warning_asked"]


def test_the_cases_cover_every_pass():
    keys = set()
    statuses = set()
    for case in CASES.values():
        for phase, entry in case["expected"]["phases"].items():
            keys.update(entry)
            statuses.add((phase, entry["status"]))
    assert {
        "stale", "tasks_total", "started_at", "manual_tasks", "manual_close_ids",
        "manual_waiting", "manual_gate", "check_findings", "batch_check_findings",
        "batch_check_accepted", "batch_check_repeats", "notice", "scope_blocker",
        "scope_deferred", "warning",
    } <= keys
    for phase in ("apply-suggestions", "apply-task-suggestions", "apply-code-fixes"):
        assert (phase, "skipped") in statuses
        assert (phase, "in_progress") in statuses
