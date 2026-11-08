"""Both write legs leave the same ``state.json`` for the same run
(docs/development/status.md, "Status parity across execution paths").

Each implement scenario runs twice from the same starting state: through the
executor's scheduler (the native invoker simulated at its dispatch seam), and
through the native leg's ``util batch-done``, recording each batch as a host
reading the same replies would.

* One run: a code task applied with its own check passing, a check task that
  reported problems, a code task applied while its own check fails, and a
  task that failed.
* A resume after a check accepted with its findings (``--human-task
  T002=done``, on record in the starting state): a code task released by it
  whose own check reports the accepted finding again, and a finding a check
  of this run reports after it (moved, the check recording second); a code
  task whose own check repeats that check's finding (split, the code task
  recording second). So the accepted and repeated findings are non-empty on
  both legs.

Every top-level key the two legs leave must be equal, or listed in
:data:`ONE_LEG_KEYS` with the reason only one leg writes it; so a new
per-task fact one leg forgets fails here, not in the next live run. The ledger
``status`` reads (``task_ledger.classify``) and the tally it prints must be
equal too, and the executor's own records must agree
(``BatchLedger.disagreements``).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

from utils import batch_done, task_ledger  # noqa: E402
from utils.batch_ledger import BatchLedger  # noqa: E402
from utils.human_task_plan import resolve_human_task_plan  # noqa: E402
from utils.output_handler import get_phase_dir  # noqa: E402
from utils.state_manager import (  # noqa: E402
    StateManager,
    set_task_check_findings_in_state,
)
from utils.status_native import implement_tally_text, implement_task_tally  # noqa: E402
from utils.subagent_tasks import PRIOR_CHANGES_TOKEN  # noqa: E402
from utils.write_phase_bridge import (  # noqa: E402
    materialize_write_tasks,
    write_task_file_for_phase,
)

from tests.test_write_executor import env  # noqa: E402,F401

PROMPT = "# Implement\n\n" + PRIOR_CHANGES_TOKEN + "\n\n1. Do the task.\n"

#: The top-level keys only one leg writes, and why
#: (docs/development/write-scheduler.md, "One record writer for both legs":
#: what only one leg does stays with that leg).
ONE_LEG_KEYS = {
    "metrics": "the executor times each batch it dispatched; a host's own "
               "`util metrics record` calls are not part of a batch record",
    "metrics_progress": "the executor's `[ETA]` progress; the native leg's "
                        "is the host's `util metrics` calls",
    "phases_started": "the executor marks implement started when it dispatches "
                      "its first task; the native leg's emit marks it",
    "updated_at": "when each leg last wrote the file",
    "task_check_trees": "a check's tree: the native leg records it in state.json, "
                        "the executor on the check's verdict stamp",
}


def _task(task_id, files, deps=()):
    return {"id": task_id, "title": f"Task {task_id}", "prompt": PROMPT,
            "files_to_modify": list(files), "depends_on": list(deps),
            "executor_role": "agent"}


@dataclass
class Scenario:
    """One implement run: the batches it hands out, each task's reply and
    the `util batch-done` flags a host records it with, the edits the
    tasks make, what an earlier run left on record, and what both legs
    must end on."""

    batches: List[Dict[str, Any]]
    replies: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]
    edits: Dict[str, Tuple[str, str]]
    exit_code: int
    task_status: Dict[str, str]
    tally: str
    #: The accepted and repeated findings the run leaves in ``state.json``,
    #: by task and check.
    accepted: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    repeats: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    earlier: Dict[str, Any] = field(default_factory=dict)


ONE_RUN = Scenario(
    batches=[
        {"batch_index": 0, "can_parallelize": False, "tasks": [_task("T001", ["src/a.py"])]},
        {"batch_index": 1, "can_parallelize": False, "tasks": [_task("T002", [], ["T001"])]},
        {"batch_index": 2, "can_parallelize": False, "tasks": [_task("T003", ["src/b.py"])]},
        {"batch_index": 3, "can_parallelize": False, "tasks": [_task("T004", ["src/c.py"])]},
    ],
    replies={
        "T001": ({"status": "applied", "ids": ["T001"], "files_touched": ["src/a.py"],
                  "summary": "edited a", "check": {"passed": True, "findings": []}},
                 dict(status="completed", own_check="passed")),
        "T002": ({"status": "no-op", "ids": ["T002"], "files_touched": [],
                  "summary": "ran the lint",
                  "check": {"passed": False, "findings": ["lint fails on a.py"]}},
                 dict(status="completed", check="findings", findings=["lint fails on a.py"])),
        "T003": ({"status": "applied", "ids": ["T003"], "files_touched": ["src/b.py"],
                  "summary": "edited b",
                  "check": {"passed": False, "findings": ["test_b fails"]}},
                 dict(status="completed", own_check="findings", findings=["test_b fails"])),
        "T004": ({"status": "failed", "ids": [], "files_touched": [],
                  "summary": "could not edit c"},
                 dict(status="failed")),
    },
    edits={"T001": ("src/a.py", "a = 2\n"), "T003": ("src/b.py", "b = 2\n")},
    exit_code=3,
    task_status={"T001": "completed", "T002": "completed", "T003": "completed",
                 "T004": "failed"},
    tally=("3/4 done; completed: T001, T003; completed, its check reported "
           "problems: T002; failed: T004"),
)

#: The findings of the resume scenario: T002's, which the user accepted,
#: T004's, which this run's check reports, and T005's own.
ACCEPTED = "node scripts/check.mjs: retries must be 3 (found 2)"
REPORTED = "node scripts/e2e.mjs: login page returns 500"
OWN = "node scripts/unit.mjs: test_e expected 1 got 0"


def _own_check(*findings):
    return {"passed": False, "findings": list(findings)}


AFTER_AN_ACCEPTED_CHECK = Scenario(
    batches=[
        {"batch_index": 0, "can_parallelize": False,
         "tasks": [_task("T003", ["src/b.py"], ["T002"])]},
        {"batch_index": 1, "can_parallelize": False, "tasks": [_task("T004", [])]},
        {"batch_index": 2, "can_parallelize": False, "tasks": [_task("T005", ["src/c.py"])]},
        {"batch_index": 3, "can_parallelize": False, "tasks": [_task("T006", ["src/a.py"])]},
    ],
    replies={
        # Released by T002's acceptance: its own check reports T002's
        # accepted finding again, and the one T004 reports after it.
        "T003": ({"status": "applied", "ids": ["T003"], "files_touched": ["src/b.py"],
                  "summary": "edited b", "check": _own_check(ACCEPTED, REPORTED)},
                 dict(status="completed", own_check="findings",
                      findings=[ACCEPTED, REPORTED])),
        "T004": ({"status": "no-op", "ids": ["T004"], "files_touched": [],
                  "summary": "ran the e2e check",
                  "check": {"passed": False, "findings": [REPORTED]}},
                 dict(status="completed", check="findings", findings=[REPORTED])),
        # After T004: its own check repeats T004's finding, beside its own.
        "T005": ({"status": "applied", "ids": ["T005"], "files_touched": ["src/c.py"],
                  "summary": "edited c", "check": _own_check(REPORTED, OWN)},
                 dict(status="completed", own_check="findings", findings=[REPORTED, OWN])),
        # A stop, so both legs end on the batch records alone: a run that
        # finishes also finalizes the phase (`phases_completed`), a step
        # outside any batch's record.
        "T006": ({"status": "failed", "ids": [], "files_touched": [],
                  "summary": "could not edit a"},
                 dict(status="failed")),
    },
    edits={"T003": ("src/b.py", "b = 2\n"), "T005": ("src/c.py", "c = 2\n")},
    exit_code=3,
    task_status={"T001": "completed", "T002": "completed", "T003": "completed",
                 "T004": "completed", "T005": "completed", "T006": "failed"},
    tally=("5/6 done; completed: T001, T003, T005; completed, its check reported "
           "problems: T004; accepted with its check findings: T002; failed: T006"),
    accepted={"T003": {"T002": [ACCEPTED]}},
    repeats={"T003": {"T004": [REPORTED]}, "T005": {"T004": [REPORTED]}},
    earlier={"T001": "completed", "T002": "accepted", "findings": {"T002": [ACCEPTED]}},
)

def _accepted_reason(task_id):
    """The reason ``--human-task <id>=done`` records for a check that
    reported problems (``resolve_human_task_plan``'s own sentence)."""
    plan = resolve_human_task_plan(
        [{"task_id": task_id}], strategy=None, decisions={task_id: "done"},
        check_findings_ids=[task_id],
    )
    return plan.accepted[task_id]


def _start(env, scenario):
    """The implement phase as its emit leaves it: the task file, and
    ``state.json`` with the decomposition, what an earlier run recorded and
    every task of this run pending."""
    (env.root / "src" / "c.py").write_text("c = 1\n", encoding="utf-8")
    phase_dir = Path(get_phase_dir(env.plan, "implement"))
    phase_dir.mkdir(parents=True, exist_ok=True)
    tasks = materialize_write_tasks(
        kind="implement", batches=scenario.batches, phase_dir=str(phase_dir),
        plan_file=str(env.plan),
    )
    write_task_file_for_phase(
        kind="implement", tasks=tasks, phase_dir=str(phase_dir), plan_file=str(env.plan),
    )
    state = StateManager(env.plan)
    state.set("implement_task_ids", list(scenario.task_status))
    earlier = dict(scenario.earlier)
    findings = earlier.pop("findings", {})
    for task_id, how in earlier.items():
        if how == "accepted":
            state.update_task_status(task_id, "completed", _accepted_reason(task_id))
        else:
            state.update_task_status(task_id, how)
    for task_id, found in findings.items():
        set_task_check_findings_in_state(state.state, task_id, found)
    for task_id in scenario.replies:
        state.update_task_status(task_id, "pending")
    state.save()
    sources = {name: (env.root / name).read_text(encoding="utf-8")
               for name in ("src/a.py", "src/b.py", "src/c.py")}
    return phase_dir, tasks, Path(state.state_file), state_text(state), sources


def state_text(state):
    return Path(state.state_file).read_text(encoding="utf-8")


def _executor_leg(env, scenario, phase_dir, tasks):
    def respond(task, prompt):
        task_id = task["task_id"]
        if task_id in scenario.edits:
            name, text = scenario.edits[task_id]
            (env.root / name).write_text(text, encoding="utf-8")
        return scenario.replies[task_id][0]

    state = StateManager(env.plan)
    result = env.scheduler(
        tasks, kind="implement", phase_dir=str(phase_dir),
        output_dir=str(phase_dir.parent), respond=respond, state=state,
    ).run()
    assert result.exit_code == scenario.exit_code, result.messages
    assert [o.task_id for o in result.outcomes] == list(scenario.replies)
    return json.loads(Path(state.state_file).read_text(encoding="utf-8"))


def _native_leg(env, scenario, phase_dir, state_file, reasons):
    for task_id, (_reply, flags) in scenario.replies.items():
        if task_id in scenario.edits:
            name, text = scenario.edits[task_id]
            (env.root / name).write_text(text, encoding="utf-8")
        if flags["status"] == "failed":
            flags = dict(flags, reason=reasons[task_id])
        batch_done.record_batch(phase_dir, task_id, state_file=state_file, **flags)
    return json.loads(state_file.read_text(encoding="utf-8"))


def _ledger(env, recorded, state_file):
    state_file.write_text(json.dumps(recorded, indent=2), encoding="utf-8")
    return (
        task_ledger.classify(StateManager(env.plan, read_only=True)),
        implement_tally_text(implement_task_tally(env.plan)),
    )


@pytest.mark.parametrize("scenario", [
    pytest.param(ONE_RUN, id="one-run"),
    pytest.param(AFTER_AN_ACCEPTED_CHECK, id="after-an-accepted-check"),
])
def test_both_legs_leave_the_same_state(env, scenario):
    phase_dir, tasks, state_file, start, sources = _start(env, scenario)
    executor = _executor_leg(env, scenario, phase_dir, tasks)
    # The executor's own records agree with each other first (WRITE-7).
    assert BatchLedger(phase_dir).disagreements(executor) == []

    state_file.write_text(start, encoding="utf-8")
    for name, text in sources.items():
        (env.root / name).write_text(text, encoding="utf-8")
    native = _native_leg(
        env, scenario, phase_dir, state_file, executor.get("task_status_reasons") or {})

    differing = sorted(
        key for key in set(executor) | set(native)
        if executor.get(key) != native.get(key)
    )
    unexplained = [key for key in differing if key not in ONE_LEG_KEYS]
    assert unexplained == [], {
        key: {"executor": executor.get(key), "native": native.get(key)}
        for key in unexplained
    }
    # Every row is a key this scenario's legs really write differently.
    assert sorted(ONE_LEG_KEYS) == [key for key in differing if key in ONE_LEG_KEYS]
    assert executor["task_status"] == scenario.task_status
    # The facts the scenario exists for are on record, on both legs (equal
    # above), so a leg that drops them fails here and not only by parity.
    assert executor.get("task_batch_check_accepted", {}) == scenario.accepted
    assert executor.get("task_batch_check_repeats", {}) == scenario.repeats

    ledger_a, tally_a = _ledger(env, executor, state_file)
    ledger_b, tally_b = _ledger(env, native, state_file)
    assert ledger_a == ledger_b
    assert tally_a == tally_b == scenario.tally
