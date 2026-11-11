"""The decisions ``implement_orchestrator.main`` takes, as tables.

Each function here is pure (no file, no print, no state), so every case is one
named row. To learn what ``implement`` does with a flag combination, find its
row; to change a rule, change its row first.
"""

from typing import Any, Dict, List

import pytest

from implement_orchestrator import (
    NATIVE_STEPS_HEADING,
    ApplyGate,
    GateAction,
    RunMode,
    StartOver,
    _applicable_task_decisions,
    closing_steps,
    decide_apply_gate,
    decide_run_mode,
    decide_tasks_missing,
    native_gate_applies,
    refuse_human_answers,
    start_over_when,
)


# --- decide_run_mode: resume, continue, or start over ------------------------

def _never_asked():
    raise AssertionError("the records were read although a flag decided")


RUN_MODE_ROWS = [
    # (id, flags, started-and-unfinished ids or _never_asked, expected RunMode)
    ("typed --resume resumes and never reads the records",
     dict(resume=True), _never_asked,
     RunMode(resume=True, resume_typed=True, start_over=False)),
    ("a plain run on an unfinished implement continues it as --resume",
     dict(), ["T001", "T002"],
     RunMode(resume=True, resume_typed=False, start_over=False,
             continued_ids=("T001", "T002"))),
    ("a plain run with nothing to keep starts over",
     dict(), [],
     RunMode(resume=False, resume_typed=False, start_over=True)),
    ("--fresh starts over and never continues",
     dict(fresh=True), _never_asked,
     RunMode(resume=False, resume_typed=False, start_over=True)),
    ("--resume wins over --fresh",
     dict(resume=True, fresh=True), _never_asked,
     RunMode(resume=True, resume_typed=True, start_over=False)),
    ("--task reruns what it names: no continue, no start-over",
     dict(task="T002"), _never_asked,
     RunMode(resume=False, resume_typed=False, start_over=False)),
    ("--fresh --task reruns the task, not every task",
     dict(task="T002", fresh=True), _never_asked,
     RunMode(resume=False, resume_typed=False, start_over=False)),
    ("--resume --task keeps the typed resume",
     dict(resume=True, task="T002"), _never_asked,
     RunMode(resume=True, resume_typed=True, start_over=False)),
    ("--dry-run on a first implement never starts over",
     dict(dry_run=True), [],
     RunMode(resume=False, resume_typed=False, start_over=False)),
    ("--dry-run on an unfinished implement previews the continue",
     dict(dry_run=True), ["T001"],
     RunMode(resume=True, resume_typed=False, start_over=False,
             continued_ids=("T001",))),
    ("--dry-run-subagents on a first implement never starts over",
     dict(dry_run_subagents=True), [],
     RunMode(resume=False, resume_typed=False, start_over=False)),
    ("--fresh --dry-run never starts over",
     dict(fresh=True, dry_run=True), _never_asked,
     RunMode(resume=False, resume_typed=False, start_over=False)),
]


@pytest.mark.parametrize(
    "flags,started,expected", [row[1:] for row in RUN_MODE_ROWS],
    ids=[row[0] for row in RUN_MODE_ROWS],
)
def test_decide_run_mode(flags, started, expected):
    kwargs = dict(resume=False, task=None, fresh=False, dry_run=False,
                  dry_run_subagents=False)
    kwargs.update(flags)
    asked = started if callable(started) else (lambda: started)
    assert decide_run_mode(started_unfinished=asked, **kwargs) == expected


# --- decide_apply_gate: D6 row 6 ---------------------------------------------

MET = {"met": True}
UNAPPLIED = {"met": False, "reason": "2 valid suggestions not yet applied",
             "valid_count": 2, "importance_breakdown": {"HIGH": 2}}
APPLY_BLOCKED = {"met": False, "reason": "scope blocker", "valid_count": 1,
                 "importance_breakdown": {}, "scope_blocked": True}
TASK_UNAPPLIED = {"met": False, "reason": "1 task suggestion", "scope_blocked": False}
TASK_BLOCKED = {"met": False, "reason": "task scope blocker", "valid_count": 0,
                "scope_blocked": True}

APPLY_GATE_ROWS = [
    # (id, apply check, task gate (None: not read), skip, yes, action, phase, refused skip)
    ("met with the task gate met proceeds",
     MET, MET, False, False, GateAction.PROCEED, "apply-suggestions", False),
    ("unapplied task suggestions are row 7's advisory, not this gate",
     MET, TASK_UNAPPLIED, False, True, GateAction.PROCEED, "apply-suggestions", False),
    ("unapplied suggestions, attended: the marker and exit 0",
     UNAPPLIED, None, False, False, GateAction.MARKER_EXIT_0, "apply-suggestions", False),
    ("unapplied suggestions under --yes: exit 3",
     UNAPPLIED, None, False, True, GateAction.NEEDS_HUMAN, "apply-suggestions", False),
    ("unapplied suggestions with --skip-prerequisites: recorded skip",
     UNAPPLIED, None, True, False, GateAction.RECORD_SKIP, "apply-suggestions", False),
    ("--skip-prerequisites wins over --yes",
     UNAPPLIED, None, True, True, GateAction.RECORD_SKIP, "apply-suggestions", False),
    ("an apply-suggestions scope blocker, attended: the marker",
     APPLY_BLOCKED, None, False, False, GateAction.MARKER_EXIT_0, "apply-suggestions", False),
    ("an apply-suggestions scope blocker under --yes: exit 3",
     APPLY_BLOCKED, None, False, True, GateAction.NEEDS_HUMAN, "apply-suggestions", False),
    ("--skip-prerequisites cannot skip a scope blocker",
     APPLY_BLOCKED, None, True, False, GateAction.NEEDS_HUMAN, "apply-suggestions", True),
    ("a task-suggestions scope blocker gates, attended: the marker",
     MET, TASK_BLOCKED, False, False, GateAction.MARKER_EXIT_0, "apply-task-suggestions", False),
    ("a task-suggestions scope blocker under --yes: exit 3",
     MET, TASK_BLOCKED, False, True, GateAction.NEEDS_HUMAN, "apply-task-suggestions", False),
    ("--skip-prerequisites cannot skip a task-suggestions scope blocker",
     MET, TASK_BLOCKED, True, True, GateAction.NEEDS_HUMAN, "apply-task-suggestions", True),
]


@pytest.mark.parametrize(
    "apply_check,task_gate,skip,yes,action,phase,refused",
    [row[1:] for row in APPLY_GATE_ROWS], ids=[row[0] for row in APPLY_GATE_ROWS],
)
def test_decide_apply_gate(apply_check, task_gate, skip, yes, action, phase, refused):
    gate = decide_apply_gate(apply_check, task_gate, skip_prerequisites=skip, unattended=yes)
    assert (gate.action, gate.gate_phase, gate.refused_skip) == (action, phase, refused)
    if action is GateAction.PROCEED:
        assert gate.missing == []
    else:
        [missing] = gate.missing
        assert missing["phase"] == phase


def test_decide_apply_gate_missing_payload():
    """The ``[PREREQUISITE_UNMET]`` entry: a scope blocker says so, and the
    counts default when the check leaves them out."""
    assert decide_apply_gate(UNAPPLIED, None, skip_prerequisites=False, unattended=True) == (
        ApplyGate(GateAction.NEEDS_HUMAN, "apply-suggestions", [{
            "phase": "apply-suggestions",
            "reason": "2 valid suggestions not yet applied",
            "valid_count": 2,
            "importance_breakdown": {"HIGH": 2},
        }])
    )
    [missing] = decide_apply_gate(
        MET, {"met": False, "reason": "r", "scope_blocked": True},
        skip_prerequisites=False, unattended=True,
    ).missing
    assert missing == {"phase": "apply-task-suggestions", "reason": "r", "valid_count": 0,
                       "importance_breakdown": {}, "scope_blocked": True}


# --- decide_tasks_missing ----------------------------------------------------

TASKS_MISSING_ROWS = [
    ("generated tasks, attended: proceed", True, False, GateAction.PROCEED),
    ("generated tasks under --yes: proceed", True, True, GateAction.PROCEED),
    ("no tasks, attended: [TASKS_MISSING] and exit 0", False, False, GateAction.MARKER_EXIT_0),
    ("no tasks under --yes: exit 3, never a silent done", False, True, GateAction.NEEDS_HUMAN),
]


@pytest.mark.parametrize(
    "completed,yes,expected", [row[1:] for row in TASKS_MISSING_ROWS],
    ids=[row[0] for row in TASKS_MISSING_ROWS],
)
def test_decide_tasks_missing(completed, yes, expected):
    assert decide_tasks_missing(generate_tasks_completed=completed, unattended=yes) is expected


# --- refuse_human_answers: D6 row 9 ------------------------------------------

KNOWN = ["T001", "T002", "T003", "H001", "H002"]
HUMAN = ["H001", "H002"]

ANSWER_ROWS = [
    # (id, answers, failed or held, typed --resume, refusal prefix or None)
    ("a human task answered done", {"H001": "done"}, set(), False, None),
    ("a human task answered skip", {"H002": "skip"}, set(), False, None),
    ("a failed code task answered on --resume", {"T002": "done"}, {"T002"}, True, None),
    ("a failed code task answered retry on --resume", {"T002": "retry"}, {"T002"}, True, None),
    ("a check holding back later tasks answered on --resume",
     {"T001": "skip"}, {"T001"}, True, None),
    ("an id not in the plan",
     {"T999": "done"}, set(), True,
     "ERROR: --human-task names task(s) that are not in this plan: T999."),
    ("an id not in the plan is refused before a missing --resume",
     {"T999": "done", "T002": "done"}, {"T002"}, False,
     "ERROR: --human-task names task(s) that are not in this plan"),
    ("a code task without --resume",
     {"T002": "done"}, {"T002"}, False,
     "ERROR: --human-task answers a code task only with --resume"),
    ("a code task that did not fail",
     {"T003": "done"}, {"T002"}, True,
     "ERROR: --human-task answers a human task, a code task the last run "
     "recorded failed, or a check whose findings hold back the tasks after it; "
     "these are code tasks that did not fail: T003."),
    ("a human task answered retry",
     {"H001": "retry"}, set(), True,
     "ERROR: --human-task <id>=retry re-runs a failed code task; a human task "
     "is answered done or skip: H001."),
    ("a missing --resume is refused before a task that did not fail",
     {"T003": "done", "H001": "retry"}, set(), False,
     "ERROR: --human-task answers a code task only with --resume"),
]


@pytest.mark.parametrize(
    "answers,failed,resume_typed,refusal", [row[1:] for row in ANSWER_ROWS],
    ids=[row[0] for row in ANSWER_ROWS],
)
def test_refuse_human_answers(answers, failed, resume_typed, refusal):
    got = refuse_human_answers(
        answers, known_ids=KNOWN, human_ids=HUMAN, failed_or_held=failed,
        resume_typed=resume_typed,
    )
    if refusal is None:
        assert got is None
    else:
        assert got is not None and got.startswith(refusal), got


def test_an_unknown_id_names_the_human_tasks_or_none():
    assert refuse_human_answers(
        {"X": "done"}, known_ids=["T001"], human_ids=[], failed_or_held=(),
        resume_typed=True,
    ).endswith("Human tasks in this plan: none")
    assert refuse_human_answers(
        {"X": "done"}, known_ids=KNOWN, human_ids=HUMAN, failed_or_held=(),
        resume_typed=True,
    ).endswith("Human tasks in this plan: H001, H002")


# --- start_over_when ---------------------------------------------------------

START_OVER_ROWS = [
    # (id, start_over, executor, --detach, --internal-worker, expected)
    ("no start-over, native", False, False, False, False, StartOver.NEVER),
    ("no start-over, executor", False, True, False, False, StartOver.NEVER),
    ("native foreground: before the emit reads the records",
     True, False, False, False, StartOver.BEFORE_EMIT),
    ("native --detach launcher: never, it may re-attach to a live run",
     True, False, True, False, StartOver.NEVER),
    ("native detached worker: before the emit",
     True, False, True, True, StartOver.BEFORE_EMIT),
    ("native worker without --detach: before the emit",
     True, False, False, True, StartOver.BEFORE_EMIT),
    ("executor foreground: past the detach boundary",
     True, True, False, False, StartOver.AFTER_DETACH),
    ("executor detached worker: past the detach boundary",
     True, True, True, True, StartOver.AFTER_DETACH),
    ("executor --detach launcher: never",
     True, True, True, False, StartOver.NEVER),
]


@pytest.mark.parametrize(
    "start_over,executor,detach,worker,expected", [row[1:] for row in START_OVER_ROWS],
    ids=[row[0] for row in START_OVER_ROWS],
)
def test_start_over_when(start_over, executor, detach, worker, expected):
    assert start_over_when(
        start_over=start_over, executor_backed=executor, detach=detach,
        internal_worker=worker,
    ) is expected


# --- native_gate_applies -----------------------------------------------------

NOTHING_WITHHELD = {"held": {}, "refused": {}, "not_run": {}}
STOP_FACTS: Dict[str, Any] = dict(
    withheld=NOTHING_WITHHELD, rerun_task=False, has_human_plan=True,
    kept_checks=False, tasks_to_dispatch=0, implement_completed=False,
)

NATIVE_GATE_ROWS = [
    ("a native emit handing out nothing beside a manual task: the stop", {}, True),
    ("the executor leg stops on its own", dict(withheld=None), False),
    ("a --task rerun", dict(rerun_task=True), False),
    ("no human-task plan", dict(has_human_plan=False), False),
    ("a task held back by a check", dict(withheld={**NOTHING_WITHHELD, "held": {"T3": "r"}}), False),
    ("a task refused before dispatch", dict(withheld={**NOTHING_WITHHELD, "refused": {"T3": "r"}}),
     False),
    ("a check kept with its findings", dict(kept_checks=True), False),
    ("a task to dispatch", dict(tasks_to_dispatch=1), False),
    ("implement already completed", dict(implement_completed=True), False),
]


@pytest.mark.parametrize(
    "change,expected", [row[1:] for row in NATIVE_GATE_ROWS],
    ids=[row[0] for row in NATIVE_GATE_ROWS],
)
def test_native_gate_applies(change, expected):
    assert native_gate_applies(**{**STOP_FACTS, **change}) is expected


# --- closing_steps -----------------------------------------------------------

CLOSING_ROWS: List[Any] = [
    # (id, closes, stopped short, must contain, must not contain)
    ("an answers-only emit on a completed implement asks for no summary and no close",
     True, False,
     ["Nothing is left to run: implement stays completed",
      "write no implementation summary", "run no `util finalize-tracking`",
      "End the report with the command `util next-command` prints"],
     ["Read the output JSON file", "generate the implementation summary",
      "--mark-completed"]),
    ("closing wins over stopping short",
     True, True,
     ["Nothing is left to run"], ["Summary file:", "--mark-completed"]),
    ("an emit that stopped short closes without --mark-completed",
     False, True,
     ["Read the output JSON file", "Summary file: /s.md",
      "without --mark-completed (tasks listed above did not run)"],
     ["Nothing is left to run", "once every task ran"]),
    ("a full emit closes with --mark-completed once every task ran",
     False, False,
     ["Read the output JSON file", "Summary file: /s.md",
      "`util finalize-tracking … --mark-completed` once every task ran"],
     ["Nothing is left to run", "(tasks listed above did not run)"]),
]


@pytest.mark.parametrize(
    "closes,stopped_short,present,absent", [row[1:] for row in CLOSING_ROWS],
    ids=[row[0] for row in CLOSING_ROWS],
)
def test_closing_steps(closes, stopped_short, present, absent):
    text = "\n".join(closing_steps(closes=closes, stopped_short=stopped_short,
                                   summary_file="/s.md"))
    assert text.count(NATIVE_STEPS_HEADING) == 1
    for words in present:
        assert words in text
    for words in absent:
        assert words not in text


def test_the_closing_steps_name_no_host():
    """The core's own words for the host (instructions/), never one host's name."""
    for closes in (True, False):
        for stopped_short in (True, False):
            text = "\n".join(closing_steps(
                closes=closes, stopped_short=stopped_short, summary_file="s"))
            assert "Claude" not in text and "Codex" not in text


# --- _applicable_task_decisions ----------------------------------------------

APPLICABLE_ROWS: List[Any] = [
    # (id, recorded, statuses, held_by, this run, --task, expected)
    ("a human task's recorded answer applies whatever its status",
     {"H1": "done"}, {"H1": "skipped"}, (), {}, None, {"H1": "done"}),
    ("a failed code task's recorded answer applies",
     {"T2": "done"}, {"T2": "failed"}, (), {}, None, {"T2": "done"}),
    ("a code task that ran again drops its recorded done",
     {"T2": "done"}, {"T2": "completed"}, (), {}, None, {}),
    ("a recorded skip keeps a skipped code task skipped",
     {"T2": "skip"}, {"T2": "skipped"}, (), {}, None, {"T2": "skip"}),
    ("a recorded done does not apply to a skipped code task",
     {"T2": "done"}, {"T2": "skipped"}, (), {}, None, {}),
    ("a check holding back later tasks is answered like a failed one",
     {"T1": "done"}, {"T1": "completed"}, ("T1",), {}, None, {"T1": "done"}),
    ("this run's retry applies to a code task, one-shot",
     {}, {"T2": "failed"}, (), {"T2": "retry"}, None, {"T2": "retry"}),
    ("this run's retry never applies to a human task",
     {}, {}, (), {"H1": "retry"}, None, {}),
    ("this run's done reaches a code task only through the record",
     {}, {"T2": "failed"}, (), {"T2": "done"}, None, {}),
    ("--task drops the answer an earlier run recorded for its task",
     {"T2": "skip"}, {"T2": "skipped"}, (), {}, "T2", {}),
    ("--task keeps an answer typed with it",
     {"T2": "skip"}, {"T2": "skipped"}, (), {"T2": "retry"}, "T2", {"T2": "retry"}),
    ("--task on a human task keeps its answer",
     {"H1": "done"}, {}, (), {}, "H1", {"H1": "done"}),
]


@pytest.mark.parametrize(
    "recorded,statuses,held_by,this_run,rerun_task,expected",
    [row[1:] for row in APPLICABLE_ROWS], ids=[row[0] for row in APPLICABLE_ROWS],
)
def test_applicable_task_decisions(recorded, statuses, held_by, this_run, rerun_task,
                                   expected):
    got = _applicable_task_decisions(
        recorded, statuses, held_by, ["H1"], this_run, rerun_task=rerun_task,
    )
    assert got == expected


def test_applicable_task_decisions_leaves_its_inputs_alone():
    recorded: Dict[str, str] = {"T2": "skip"}
    _applicable_task_decisions(recorded, {}, (), [], {}, rerun_task="T2")
    assert recorded == {"T2": "skip"}
