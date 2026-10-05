"""`utils/next_phase.py`: the one builder of the next-phase command (T3).

plans/next-phase-command.md §2.1 is the contract; §5 criterion 1 ("one string
per route") is the table test below. Every row runs on all six host/route rows
of the §2.1 table against a real ``state.json`` (built with ``StateManager``)
plus the ``grouped.json`` / ``validation.json`` files the workflow view reads,
and pins the literal command each route must print. The prefixes are typed out
here on purpose, not read from ``hosts/capabilities.yaml``: a change to a
template has to change this table too.
"""

import io
import json
import os
import shlex
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import check_workflow_prerequisites as prereq  # noqa: E402
import modes  # noqa: E402
from utils import cli_common  # noqa: E402
from utils import next_phase as np_  # noqa: E402
from utils.next_phase import NextPhase, OptionalStep  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

#: The six host/route rows of the §2.1 table, with the literal text before
#: ``{args}``.
ROUTES = [
    ("claude-code", None, "/multi-llm:multi-llm "),
    ("claude-code", "skill", "/multi-llm "),
    ("codex", None, "Use the multi-llm skill with exactly these arguments: "),
    ("codex", "package", "$multi-llm:multi-llm "),
    ("opencode", None, "/multi-llm "),
    ("opencode", "package", "/multi-llm "),
]
ROUTE_IDS = [f"{host}.{route or 'default'}" for host, route, _ in ROUTES]

P = "plans/foo.md"

WHY_REVIEW_TASKS = "optional: have the models review the tasks before implementing"
WHY_APPLY_TASKS = "optional: apply the task-review suggestions before implementing"
NOTICE_ONE_GROUP = (
    "review-tasks produced 1 unapplied task suggestion groups. Run "
    "--apply-task-suggestions to apply them, or proceed to --implement to skip."
)

# --- fixture files --------------------------------------------------------

VALIDATION_VALID = {"format_version": 2, "groups": [
    {"group_index": 0, "status": "valid"},
]}
VALIDATION_ALL_INVALID = {"format_version": 2, "groups": [
    {"group_index": 0, "status": "invalid"},
    {"group_index": 1, "status": "invalid"},
]}
ONE_GROUP = {"format_version": 2, "groups": [
    {"group_id": "g1", "theme": "t",
     "suggestions": [{"title": "s", "importance": "HIGH"}]},
]}
NO_FINDINGS = {"format_version": 2, "groups": [{"group_id": "g1", "suggestions": []}]}

RP_VALID = {("review-plan", "validation.json"): VALIDATION_VALID}
RP_ALL_INVALID = {("review-plan", "validation.json"): VALIDATION_ALL_INVALID}
RT_ACTIONABLE = {
    ("review-tasks", "grouped.json"): ONE_GROUP,
    ("review-tasks", "validation.json"): VALIDATION_VALID,
}
RT_CLEAN = {("review-tasks", "grouped.json"): NO_FINDINGS}
CR_FINDING = {("code-review", "grouped.json"): ONE_GROUP}
CR_CLEAN = {("code-review", "grouped.json"): NO_FINDINGS}

UP_TO_TASKS = ("review-plan", "apply-suggestions", "generate-tasks")
UP_TO_REVIEW_TASKS = UP_TO_TASKS + ("review-tasks",)
UP_TO_IMPLEMENT = UP_TO_TASKS + ("implement",)
UP_TO_CODE_REVIEW = UP_TO_IMPLEMENT + ("code-review",)
ALL_REQUIRED = UP_TO_CODE_REVIEW + ("apply-code-fixes",)
IMPLEMENT_PARTIAL = {"T001": "completed", "T002": "pending"}


def S(completed=(), skipped=(), tasks=None, files=None):
    """A state spec: phases recorded completed / skipped, task statuses, files."""
    return {
        "completed": tuple(completed),
        "skipped": tuple(skipped),
        "tasks": dict(tasks or {}),
        "files": dict(files or {}),
    }


def _merge(*file_maps):
    merged = {}
    for file_map in file_maps:
        merged.update(file_map)
    return merged


def _build(tmp_path: Path, spec) -> Path:
    plan = tmp_path / "plans" / "foo.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("# foo\n", encoding="utf-8")
    for (phase_dir, name), content in spec["files"].items():
        target = output_dir_path(plan) / phase_dir
        target.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else json.dumps(content)
        (target / name).write_text(text, encoding="utf-8")
    state = StateManager(plan)
    for phase in spec["completed"]:
        state.mark_phase_completed(phase)
    for phase in spec["skipped"]:
        state.mark_phase_skipped(phase, f"test skip of {phase}")
    for task_id, status in spec["tasks"].items():
        state.update_task_status(task_id, status)
    state.save()
    return plan


def _workflow(plan: Path) -> dict:
    return prereq.get_workflow_status(StateManager(plan), plan, output_dir_path(plan))


# --- the table ------------------------------------------------------------
#
# expect is one of:
#   None                      -> the builder answers None (no key, no block)
#   "complete"                -> NextPhase(complete=True)
#   (primary_args, [(optional_phase, optional_args, why), ...])
#       where *_args is everything after the route's prefix.

def A(primary_args, *optional):
    return (primary_args, list(optional))


ROWS = [
    # -- the forward walk ---------------------------------------------------
    ("review-plan->apply-suggestions",
     S(("review-plan",), files=RP_VALID), "review-plan", None,
     A(f"--apply-suggestions {P}")),
    ("review-plan-all-invalid->generate-tasks",
     S(("review-plan",), files=RP_ALL_INVALID), "review-plan", None,
     A(f"--generate-tasks {P}")),
    ("apply-suggestions->generate-tasks",
     S(("review-plan", "apply-suggestions"), files=RP_VALID), "apply-suggestions", None,
     A(f"--generate-tasks {P}")),
    ("generate-tasks->implement+review-tasks-only",
     S(UP_TO_TASKS), "generate-tasks", None,
     A(f"--implement {P}", ("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS))),
    ("review-tasks-actionable->implement+apply-task-suggestions-once-with-notice",
     S(UP_TO_REVIEW_TASKS, files=RT_ACTIONABLE), "review-tasks", None,
     A(f"--implement {P}",
       ("apply-task-suggestions", f"--apply-task-suggestions {P}", NOTICE_ONE_GROUP))),
    ("required-done-notice-fires->complete",
     S(UP_TO_REVIEW_TASKS + ("implement", "code-review", "apply-code-fixes"),
       files=RT_ACTIONABLE),
     "review-tasks", None, "complete"),
    ("review-tasks-no-findings->implement",
     S(UP_TO_REVIEW_TASKS, files=RT_CLEAN), "review-tasks", None,
     A(f"--implement {P}")),
    ("apply-task-suggestions->implement",
     S(UP_TO_REVIEW_TASKS + ("apply-task-suggestions",), files=RT_ACTIONABLE),
     "apply-task-suggestions", None,
     A(f"--implement {P}")),
    ("implement->review-code",
     S(UP_TO_IMPLEMENT), "implement", None,
     A(f"--review-code {P}")),
    ("review-code-findings->apply-code-fixes",
     S(UP_TO_CODE_REVIEW, files=CR_FINDING), "review-code", None,
     A(f"--apply-code-fixes {P}")),
    ("review-code-alias-code-review->apply-code-fixes",
     S(UP_TO_CODE_REVIEW, files=CR_FINDING), "code-review", None,
     A(f"--apply-code-fixes {P}")),
    ("review-code-clean->complete",
     S(UP_TO_CODE_REVIEW, files=CR_CLEAN), "review-code", None, "complete"),
    ("review-code-clean-fallback->complete",
     S(UP_TO_CODE_REVIEW, files=CR_CLEAN), None, None, "complete"),
    ("apply-code-fixes->complete",
     S(ALL_REQUIRED, files=CR_FINDING), "apply-code-fixes", None, "complete"),
    ("only-optional-left-open->complete",
     S(("generate-tasks", "implement", "code-review", "apply-code-fixes")),
     "generate-tasks", None, "complete"),
    ("full-exit-0->complete",
     S(ALL_REQUIRED + ("review-tasks",), skipped=("apply-task-suggestions",),
       files=_merge(RT_CLEAN, CR_FINDING)),
     "full", None, "complete"),
    ("full-falls-back-to-suggested_next",
     S(("review-plan",), files=RP_VALID), "full", None,
     A(f"--apply-suggestions {P}")),

    # -- --resume, built from state ----------------------------------------
    ("after-implement-in-progress->implement-resume",
     S(UP_TO_TASKS, tasks=IMPLEMENT_PARTIAL), None, "implement",
     A(f"--implement {P} --resume")),
    ("no-after-implement-in-progress->implement-resume",
     S(UP_TO_TASKS, tasks=IMPLEMENT_PARTIAL), None, None,
     A(f"--implement {P} --resume",
       ("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS))),
    ("review-tasks-while-implement-in-progress->implement-resume",
     S(UP_TO_REVIEW_TASKS, tasks=IMPLEMENT_PARTIAL, files=RT_CLEAN), "review-tasks", None,
     A(f"--implement {P} --resume")),
    ("finished-implement-in-progress->implement-resume",
     S(UP_TO_TASKS, tasks=IMPLEMENT_PARTIAL), "implement", None,
     A(f"--implement {P} --resume")),
    ("rerun-review-plan-while-apply-suggestions-in-progress",
     S(("review-plan",), tasks={"apply-batch-0": "completed"}, files=RP_VALID),
     "review-plan", None,
     A(f"--apply-suggestions {P} --resume")),
    ("after-apply-suggestions-in-progress",
     S(("review-plan",), tasks={"apply-batch-0": "completed"}, files=RP_VALID),
     None, "apply-suggestions",
     A(f"--apply-suggestions {P} --resume")),
    ("review-tasks-while-apply-task-suggestions-in-progress",
     S(UP_TO_REVIEW_TASKS, tasks={"apply-task-batch-0": "completed"}, files=RT_ACTIONABLE),
     "review-tasks", None,
     A(f"--implement {P}",
       ("apply-task-suggestions", f"--apply-task-suggestions {P} --resume",
        NOTICE_ONE_GROUP))),
    ("rerun-review-code-while-apply-code-fixes-in-progress",
     S(UP_TO_CODE_REVIEW, tasks={"apply-fix-batch-0": "completed"}, files=CR_FINDING),
     "review-code", None,
     A(f"--apply-code-fixes {P} --resume")),
    ("negative-implement-no-task-completed",
     S(UP_TO_TASKS, tasks={"T001": "pending", "T002": "pending"}), "generate-tasks", None,
     A(f"--implement {P}", ("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS))),
    ("negative-apply-suggestions-batch-failed",
     S(("review-plan",), tasks={"apply-batch-0": "failed"}, files=RP_VALID),
     "review-plan", None,
     A(f"--apply-suggestions {P}")),
    ("negative-apply-task-suggestions-batch-failed",
     S(UP_TO_REVIEW_TASKS, tasks={"apply-task-batch-0": "failed"}, files=RT_ACTIONABLE),
     "review-tasks", None,
     A(f"--implement {P}",
       ("apply-task-suggestions", f"--apply-task-suggestions {P}", NOTICE_ONE_GROUP))),
    ("negative-apply-code-fixes-batch-pending",
     S(UP_TO_CODE_REVIEW, tasks={"apply-fix-batch-0": "pending"}, files=CR_FINDING),
     "review-code", None,
     A(f"--apply-code-fixes {P}")),

    # -- a started, unfinished phase before the finished one comes first ----
    # (2026-10-02: the walk used to start after the finished phase, so the
    # `user_next_step` disagreed with the `suggested next` on the same result)
    ("apply-code-fixes-finished-while-implement-in-progress->implement-resume",
     S(UP_TO_TASKS + ("code-review", "apply-code-fixes"),
       tasks=IMPLEMENT_PARTIAL, files=CR_FINDING),
     "apply-code-fixes", None,
     A(f"--implement {P} --resume",
       ("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS))),
    ("review-code-finished-while-implement-seeded-not-run->implement",
     S(UP_TO_TASKS + ("code-review",),
       tasks={"T001": "pending", "T002": "pending"}, files=CR_FINDING),
     "review-code", None,
     A(f"--implement {P}",
       ("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS))),
    ("after-apply-code-fixes-while-implement-in-progress",
     S(UP_TO_TASKS + ("code-review", "apply-code-fixes"),
       tasks=IMPLEMENT_PARTIAL, files=CR_FINDING),
     None, "apply-code-fixes",
     A(f"--implement {P} --resume",
       ("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS))),

    # -- the gate: the finished phase is not on record ---------------------
    ("gate-review-plan-pending", S(), "review-plan", None, None),
    ("gate-apply-suggestions-pending",
     S(("review-plan",), files=RP_VALID), "apply-suggestions", None, None),
    ("gate-generate-tasks-pending",
     S(("review-plan", "apply-suggestions")), "generate-tasks", None, None),
    ("gate-review-tasks-pending-while-suggested-next-is-implement",
     S(UP_TO_TASKS), "review-tasks", None, None),
    ("gate-apply-task-suggestions-pending-native-first-call",
     S(UP_TO_REVIEW_TASKS, files=RT_ACTIONABLE), "apply-task-suggestions", None, None),
    ("gate-implement-pending", S(UP_TO_TASKS), "implement", None, None),
    ("gate-review-code-pending", S(UP_TO_IMPLEMENT), "review-code", None, None),
    ("gate-apply-code-fixes-pending-native-first-call",
     S(UP_TO_CODE_REVIEW, files=CR_FINDING), "apply-code-fixes", None, None),

    # -- no phase, no answer ------------------------------------------------
    *[
        (f"{word}->none", S(("review-plan",), files=RP_VALID), word, None, None)
        for word in ("ask", "status", "init", "doctor", "install-host",
                     "uninstall-host", "not-a-mode")
    ],

    # -- starting mid-workflow never sends the user back -------------------
    ("standalone-generate-tasks->implement+review-tasks",
     S(("generate-tasks",)), "generate-tasks", None,
     A(f"--implement {P}", ("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS))),
    ("standalone-review-code-findings->apply-code-fixes",
     S(("code-review",), files=CR_FINDING), "review-code", None,
     A(f"--apply-code-fixes {P}")),
    ("standalone-review-code-clean->complete",
     S(("code-review",), files=CR_CLEAN), "review-code", None, "complete"),
    ("standalone-review-tasks-actionable->implement+apply-task-suggestions",
     S(("review-tasks",), files=RT_ACTIONABLE), "review-tasks", None,
     A(f"--implement {P}",
       ("apply-task-suggestions", f"--apply-task-suggestions {P}", NOTICE_ONE_GROUP))),

    # -- util next-command --after: the same answers, gate off -------------
    ("after-standalone-generate-tasks",
     S(("generate-tasks",)), None, "generate-tasks",
     A(f"--implement {P}", ("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS))),
    ("after-standalone-review-code-findings",
     S(("code-review",), files=CR_FINDING), None, "review-code",
     A(f"--apply-code-fixes {P}")),
    ("after-standalone-review-code-clean",
     S(("code-review",), files=CR_CLEAN), None, "review-code", "complete"),
    ("after-standalone-review-tasks-actionable",
     S(("review-tasks",), files=RT_ACTIONABLE), None, "review-tasks",
     A(f"--implement {P}",
       ("apply-task-suggestions", f"--apply-task-suggestions {P}", NOTICE_ONE_GROUP))),
    ("after-a-pending-phase-still-walks-forward",
     S(), None, "generate-tasks",
     A(f"--implement {P}")),
    ("after-full->fallback",
     S(UP_TO_TASKS), None, "full",
     A(f"--implement {P}", ("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS))),
    ("after-ask->none", S(UP_TO_TASKS), None, "ask", None),
    ("finished-mode-wins-over-after",
     S(UP_TO_IMPLEMENT), "implement", "generate-tasks",
     A(f"--review-code {P}")),

    # -- util next-command without --after: plain suggested_next -----------
    ("no-after-standalone-generate-tasks->review-plan",
     S(("generate-tasks",)), None, None, A(f"--review-plan {P}")),
    ("no-after-standalone-review-code->review-plan",
     S(("code-review",), files=CR_FINDING), None, None, A(f"--review-plan {P}")),
    ("no-after-standalone-review-tasks->review-plan",
     S(("review-tasks",), files=RT_ACTIONABLE), None, None, A(f"--review-plan {P}")),
    ("no-after-generate-tasks-done->implement+review-tasks",
     S(UP_TO_TASKS), None, None,
     A(f"--implement {P}", ("review-tasks", f"--review-tasks {P}", WHY_REVIEW_TASKS))),
]


def _expected(expect, prefix):
    if expect is None:
        return None
    if expect == "complete":
        return {"phase": None, "command": None, "complete": True}
    primary_args, optional = expect
    primary_phase = modes.phase_for(primary_args.split()[0][2:])
    payload = {"phase": primary_phase, "command": prefix + primary_args}
    if optional:
        payload["optional_before"] = [
            {"phase": phase, "command": prefix + args, "why": why}
            for phase, args, why in optional
        ]
    payload["complete"] = False
    return payload


@pytest.mark.parametrize(
    "spec,finished,after,expect",
    [row[1:] for row in ROWS],
    ids=[row[0] for row in ROWS],
)
def test_table(tmp_path, spec, finished, after, expect):
    """§5 criterion 1: one literal string per route, for every row."""
    plan = _build(tmp_path, spec)
    workflow = _workflow(plan)
    for host, route, prefix in ROUTES:
        where = f"{host}.{route or 'default'}"
        answer = np_.next_phase_commands(
            P, host, route, finished, after_mode=after, plan_path=plan
        )
        got = None if answer is None else answer.to_dict()
        assert got == _expected(expect, prefix), where
        # A caller's precomputed payload gives the same answer, and a re-run
        # against the same state (the exit-2 case) gives the same answer too.
        again = np_.next_phase_commands(
            P, host, route, finished, after_mode=after, workflow=workflow
        )
        assert (None if again is None else again.to_dict()) == got, where


def test_the_table_covers_every_workflow_mode_and_full():
    finished = {row[2] for row in ROWS} | {row[3] for row in ROWS}
    for mode in modes.MODES:
        if mode.is_workflow_phase or mode.runs_workflow:
            assert mode.name in finished, mode.name


class TestLiteralCommands:
    """A few rows spelled out whole, so the table's prefix arithmetic cannot
    hide a wrong join."""

    def test_implement_resume_on_every_route(self, tmp_path):
        plan = _build(tmp_path, S(UP_TO_TASKS, tasks=IMPLEMENT_PARTIAL))
        expected = {
            ("claude-code", None): "/multi-llm:multi-llm --implement plans/foo.md --resume",
            ("claude-code", "skill"): "/multi-llm --implement plans/foo.md --resume",
            ("codex", None): (
                "Use the multi-llm skill with exactly these arguments: "
                "--implement plans/foo.md --resume"
            ),
            ("codex", "package"): "$multi-llm:multi-llm --implement plans/foo.md --resume",
            ("opencode", None): "/multi-llm --implement plans/foo.md --resume",
            ("opencode", "package"): "/multi-llm --implement plans/foo.md --resume",
        }
        for (host, route), command in expected.items():
            answer = np_.next_phase_commands(P, host, route, None,
                                             after_mode="implement", plan_path=plan)
            assert answer.command == command

    def test_review_code_maps_to_its_mode_word(self, tmp_path):
        plan = _build(tmp_path, S(UP_TO_IMPLEMENT))
        answer = np_.next_phase_commands(P, "claude-code", None, "implement",
                                         plan_path=plan)
        assert answer.phase == "code-review"
        assert answer.command == "/multi-llm:multi-llm --review-code plans/foo.md"

    def test_the_plan_example_json(self, tmp_path):
        """§2.1's serialised example, byte for byte (after apply-suggestions)."""
        plan = _build(tmp_path, S(UP_TO_TASKS))
        answer = np_.next_phase_commands(P, "claude-code", None, "generate-tasks",
                                         plan_path=plan)
        assert json.dumps(answer.to_dict()) == json.dumps({
            "phase": "implement",
            "command": "/multi-llm:multi-llm --implement plans/foo.md",
            "optional_before": [
                {"phase": "review-tasks",
                 "command": "/multi-llm:multi-llm --review-tasks plans/foo.md",
                 "why": "optional: have the models review the tasks before implementing"},
            ],
            "complete": False,
        })


class TestNoticeAndWalk:
    def test_the_notice_really_fires_in_the_complete_row(self, tmp_path):
        """The walk wins over the notice: it fires, and still adds nothing."""
        spec = S(UP_TO_REVIEW_TASKS + ("implement", "code-review", "apply-code-fixes"),
                 files=RT_ACTIONABLE)
        plan = _build(tmp_path, spec)
        assert _workflow(plan)["notices"] == [NOTICE_ONE_GROUP]
        answer = np_.next_phase_commands(P, "codex", "package", "review-tasks",
                                         plan_path=plan)
        assert answer.to_dict() == {"phase": None, "command": None, "complete": True}

    def test_the_gate_row_where_suggested_next_is_implement(self, tmp_path):
        """The old `suggested_next == phase` guard printed implement here."""
        plan = _build(tmp_path, S(UP_TO_TASKS))
        assert _workflow(plan)["suggested_next"] == "implement"
        assert np_.next_phase_commands(P, "claude-code", None, "review-tasks",
                                       plan_path=plan) is None

    def test_a_notice_without_a_walk_entry_adds_nothing(self):
        """A notice on a phase the walk did not list never adds an entry."""
        workflow = {
            "phases": {
                "review-plan": {"status": "completed"},
                "apply-suggestions": {"status": "completed"},
                "generate-tasks": {"status": "completed"},
                "review-tasks": {"status": "completed"},
                "apply-task-suggestions": {"status": "pending", "notice": "N"},
                "implement": {"status": "pending"},
                "code-review": {"status": "pending"},
                "apply-code-fixes": {"status": "pending"},
            },
            "suggested_next": "implement",
        }
        # Walk from implement's successor span: the optional phase is behind it.
        answer = np_.next_phase_commands(P, "claude-code", None, None,
                                         after_mode="implement", workflow=workflow)
        assert answer.phase == "code-review"
        assert answer.optional_before == ()
        # And from review-tasks, the notice supplies the why of the one entry.
        answer = np_.next_phase_commands(P, "claude-code", None, "review-tasks",
                                         workflow=workflow)
        assert [s.why for s in answer.optional_before] == ["N"]

    def test_the_generic_why_when_no_notice_fires(self):
        workflow = {
            "phases": {
                "review-plan": {"status": "completed"},
                "apply-suggestions": {"status": "completed"},
                "generate-tasks": {"status": "completed"},
                "review-tasks": {"status": "completed"},
                "apply-task-suggestions": {"status": "pending"},
                "implement": {"status": "pending"},
                "code-review": {"status": "pending"},
                "apply-code-fixes": {"status": "pending"},
            },
            "suggested_next": "implement",
        }
        answer = np_.next_phase_commands(P, "opencode", None, "review-tasks",
                                         workflow=workflow)
        assert answer.optional_before == (OptionalStep(
            "apply-task-suggestions", "/multi-llm --apply-task-suggestions plans/foo.md",
            WHY_APPLY_TASKS,
        ),)

    def test_the_builder_never_reads_the_route_from_the_environment(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(modes.ROUTE_ENV, "package")
        plan = _build(tmp_path, S(UP_TO_IMPLEMENT))
        answer = np_.next_phase_commands(P, "codex", None, "implement", plan_path=plan)
        assert answer.command == (
            "Use the multi-llm skill with exactly these arguments: "
            "--review-code plans/foo.md"
        )

    def test_a_route_the_host_has_no_template_for_raises(self, tmp_path):
        plan = _build(tmp_path, S(UP_TO_IMPLEMENT))
        with pytest.raises(ValueError, match="claude-code"):
            np_.next_phase_commands(P, "claude-code", "package", "implement",
                                    plan_path=plan)
        with pytest.raises(ValueError, match="codex"):
            np_.next_phase_commands(P, "codex", "skill", None, plan_path=plan)

    def test_plan_path_defaults_to_the_absolute_plan_arg(self, tmp_path, monkeypatch):
        plan = _build(tmp_path, S(UP_TO_IMPLEMENT))
        monkeypatch.chdir(tmp_path)
        answer = np_.next_phase_commands(P, "claude-code", None, "implement")
        assert answer.command == "/multi-llm:multi-llm --review-code plans/foo.md"
        assert plan.is_file()


# --- quoting ----------------------------------------------------------------

WINDOWS = "C:\\Users\\John Smith\\plans\\foo.md"
BOTH = "plans/it's \"x\".md"

#: (plan_arg, printed form)
QUOTING = [
    ("plans/foo.md", "plans/foo.md"),
    ("/home/u/plans/foo.md", "/home/u/plans/foo.md"),
    ("plans/my plan.md", "\"plans/my plan.md\""),
    ("plans/tab\there.md", "\"plans/tab\there.md\""),
    ("plans/it's.md", "\"plans/it's.md\""),
    ("plans/say \"hi\".md", "'plans/say \"hi\".md'"),
    (BOTH, "\"plans/it's \"x\".md\""),
    (WINDOWS, "\"C:\\Users\\John Smith\\plans\\foo.md\""),
    ("'plans/foo.md'", "\"'plans/foo.md'\""),
]
QUOTING_IDS = ["relative", "absolute", "space", "tab", "single", "double", "both",
               "windows", "already-quoted"]


@pytest.mark.parametrize("plan_arg,printed", QUOTING, ids=QUOTING_IDS)
def test_quote_plan_arg(plan_arg, printed):
    assert np_.quote_plan_arg(plan_arg) == printed


@pytest.fixture(scope="module")
def after_tasks_workflow(tmp_path_factory):
    plan = _build(tmp_path_factory.mktemp("quoting"), S(UP_TO_TASKS))
    return _workflow(plan)


@pytest.mark.parametrize("plan_arg,printed", QUOTING, ids=QUOTING_IDS)
def test_quoting_on_all_six_routes(after_tasks_workflow, plan_arg, printed):
    """One rule on every route; `plan_arg` is never resolved or rebased."""
    for host, route, prefix in ROUTES:
        answer = np_.next_phase_commands(
            plan_arg, host, route, "generate-tasks", workflow=after_tasks_workflow
        )
        assert answer.command == f"{prefix}--implement {printed}"
        assert answer.optional_before[0].command == f"{prefix}--review-tasks {printed}"


def _strip_one_matching_layer(text: str) -> str:
    """The `plan` property's documented reader: at most one layer of matching quotes."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


@pytest.mark.parametrize("plan_arg,printed", QUOTING, ids=QUOTING_IDS)
def test_host_parsing_round_trips_the_plan_arg(after_tasks_workflow, plan_arg, printed):
    """Model the documented reader, then the shell routes' `"<plan-path>"` slot."""
    for host, route, prefix in ROUTES:
        answer = np_.next_phase_commands(
            plan_arg, host, route, "generate-tasks", workflow=after_tasks_workflow
        )
        typed = answer.command[len(f"{prefix}--implement "):]
        assert typed == printed
        read = _strip_one_matching_layer(typed)
        if plan_arg == BOTH:
            continue  # no unescaped spelling exists; no route is claimed to round-trip it
        assert read == plan_arg
        if '"' in plan_arg:
            continue  # the shims' double-quoted slot cannot carry a `"` today
        dispatcher = (
            'uv run --frozen --no-dev --project "/opt/skill dir" -- python '
            f'"/opt/skill dir/multi_llm.py" implement "{read}" --host {host}'
        )
        argv = shlex.split(dispatcher)
        assert argv[argv.index("implement") + 1] == plan_arg


# --- the block --------------------------------------------------------------

class TestRenderNextPhaseLines:
    def test_primary_first_then_each_pair_in_order(self):
        answer = NextPhase(
            phase="implement",
            command="/x --implement p",
            optional_before=(
                OptionalStep("review-tasks", "/x --review-tasks p", "why one"),
                OptionalStep("apply-task-suggestions", "/x --apply-task-suggestions p",
                             "why two"),
            ),
        )
        assert np_.render_next_phase_lines(answer) == [
            "[NEXT_COMMAND] /x --implement p",
            "[NEXT_COMMAND_OPTIONAL_WHY] why one",
            "[NEXT_COMMAND_OPTIONAL] /x --review-tasks p",
            "[NEXT_COMMAND_OPTIONAL_WHY] why two",
            "[NEXT_COMMAND_OPTIONAL] /x --apply-task-suggestions p",
        ]

    def test_complete_renders_only_the_complete_marker(self):
        answer = NextPhase(phase=None, command=None, complete=True)
        assert np_.render_next_phase_lines(answer) == ["[WORKFLOW_COMPLETE]"]

    def test_no_optional_renders_one_line(self):
        answer = NextPhase(phase="review-code", command="/x --review-code p")
        assert np_.render_next_phase_lines(answer) == ["[NEXT_COMMAND] /x --review-code p"]

    def test_command_lines_carry_the_command_byte_for_byte(self, after_tasks_workflow):
        for plan_arg, _printed in QUOTING:
            answer = np_.next_phase_commands(
                plan_arg, "codex", None, "generate-tasks", workflow=after_tasks_workflow
            )
            lines = np_.render_next_phase_lines(answer)
            assert lines[0] == f"{np_.NEXT_COMMAND_MARKER} {answer.command}"
            assert lines[0][len(np_.NEXT_COMMAND_MARKER) + 1:] == answer.command
            step = answer.optional_before[0]
            assert lines[1] == f"{np_.NEXT_COMMAND_OPTIONAL_WHY_MARKER} {step.why}"
            assert lines[2][len(np_.NEXT_COMMAND_OPTIONAL_MARKER) + 1:] == step.command

    def test_marker_spellings(self):
        assert np_.NEXT_COMMAND_MARKER == "[NEXT_COMMAND]"
        assert np_.NEXT_COMMAND_OPTIONAL_WHY_MARKER == "[NEXT_COMMAND_OPTIONAL_WHY]"
        assert np_.NEXT_COMMAND_OPTIONAL_MARKER == "[NEXT_COMMAND_OPTIONAL]"
        assert np_.WORKFLOW_COMPLETE_MARKER == "[WORKFLOW_COMPLETE]"


class TestToDict:
    def test_complete_is_exactly_three_keys(self):
        assert NextPhase(phase=None, command=None, complete=True).to_dict() == {
            "phase": None, "command": None, "complete": True,
        }

    def test_optional_before_is_omitted_when_empty(self):
        payload = NextPhase(phase="implement", command="c").to_dict()
        assert list(payload) == ["phase", "command", "complete"]
        assert payload["complete"] is False

    def test_key_order_with_optional_entries(self):
        payload = NextPhase(
            phase="implement", command="c",
            optional_before=(OptionalStep("review-tasks", "o", "w"),),
        ).to_dict()
        assert list(payload) == ["phase", "command", "optional_before", "complete"]
        assert payload["optional_before"] == [
            {"phase": "review-tasks", "command": "o", "why": "w"},
        ]


# --- Mode.optional_why ------------------------------------------------------

def test_optional_why_is_set_exactly_on_the_optional_phases():
    carrying = {m.name for m in modes.MODES if m.optional_why}
    optional = {m.name for m in modes.MODES if m.phase in prereq.OPTIONAL_PHASES}
    assert carrying == optional
    by_name = {m.name: m for m in modes.MODES}
    assert by_name["review-tasks"].optional_why == WHY_REVIEW_TASKS
    assert by_name["apply-task-suggestions"].optional_why == WHY_APPLY_TASKS


# --- resolve_route ----------------------------------------------------------

def _resolve(host, env, **kwargs):
    err = io.StringIO()
    route = np_.resolve_route(host, environ=env, stderr=err, **kwargs)
    return route, [line for line in err.getvalue().splitlines() if line]


CACHE_ROOT = (
    "/home/u/.claude/plugins/cache/beastlabai/multi-llm/0123abcd/skills/multi-llm"
)


class TestResolveRoute:
    @pytest.mark.parametrize("host", ["codex", "opencode"])
    def test_unset_is_the_default_silently(self, host):
        assert _resolve(host, {}) == (None, [])

    @pytest.mark.parametrize("host", ["codex", "opencode"])
    def test_the_tool_route_is_honoured(self, host):
        value = modes.TOOL_ROUTES[0]
        assert _resolve(host, {modes.ROUTE_ENV: value}) == (value, [])

    @pytest.mark.parametrize("host", ["codex", "opencode"])
    @pytest.mark.parametrize("value", ["", "pakage", "skill", "plugin", "default"])
    def test_an_invalid_value_is_the_default_plus_one_line(self, host, value):
        route, lines = _resolve(host, {modes.ROUTE_ENV: value})
        assert route is None
        assert len(lines) == 1
        assert repr(value) in lines[0] and modes.ROUTE_ENV in lines[0]

    def test_a_tool_route_member_with_no_cell_is_the_default(self, monkeypatch):
        monkeypatch.setattr(modes, "TOOL_ROUTES", modes.TOOL_ROUTES + ("nightly",))
        route, lines = _resolve("codex", {modes.ROUTE_ENV: "nightly"})
        assert route is None
        assert len(lines) == 1 and "'nightly'" in lines[0]

    def test_package_under_claude_code_is_ignored_with_one_line(self):
        value = modes.TOOL_ROUTES[0]
        route, lines = _resolve("claude-code", {modes.ROUTE_ENV: value},
                                skill_root=CACHE_ROOT)
        assert route is None
        assert len(lines) == 1 and repr(value) in lines[0]
        route, lines = _resolve("claude-code", {modes.ROUTE_ENV: value},
                                skill_root="/p/.claude/skills/multi-llm")
        assert route == "skill"
        assert len(lines) == 1

    def test_claude_code_reads_no_env_silently(self):
        assert _resolve("claude-code", {}, skill_root=CACHE_ROOT) == (None, [])

    def test_project_skill_root_is_the_skill_route(self, tmp_path):
        root = tmp_path / "proj" / ".claude" / "skills" / "multi-llm"
        assert _resolve("claude-code", {}, skill_root=root) == ("skill", [])

    def test_home_skill_root_is_the_skill_route(self):
        root = Path.home() / ".claude" / "skills" / "multi-llm"
        assert _resolve("claude-code", {}, skill_root=root) == ("skill", [])
        assert _resolve("claude-code", {}, skill_root=str(root) + "/") == ("skill", [])

    def test_windows_separators(self):
        root = "C:\\Users\\me\\.claude\\skills\\multi-llm"
        assert _resolve("claude-code", {}, skill_root=root) == ("skill", [])
        cache = "C:\\Users\\me\\.claude\\plugins\\cache\\beastlabai\\multi-llm\\ab\\skills\\multi-llm"
        assert _resolve("claude-code", {}, skill_root=cache) == (None, [])

    def test_plugin_cache_and_dev_checkout_are_the_default(self):
        assert _resolve("claude-code", {}, skill_root=CACHE_ROOT) == (None, [])
        assert _resolve("claude-code", {},
                        skill_root="/home/u/code/multi-llm-plugin/skills/multi-llm") == (None, [])
        # A partial tail is not an install path.
        assert _resolve("claude-code", {}, skill_root="/x/skills/multi-llm") == (None, [])

    def test_a_symlink_is_judged_by_its_own_path(self, tmp_path):
        cache = tmp_path / "home" / ".claude" / "plugins" / "cache" / "beastlabai" \
            / "multi-llm" / "0123abcd" / "skills" / "multi-llm"
        cache.mkdir(parents=True)
        link = tmp_path / "proj" / ".claude" / "skills" / "multi-llm"
        link.parent.mkdir(parents=True)
        installed = tmp_path / "elsewhere" / ".claude" / "skills" / "multi-llm"
        installed.mkdir(parents=True)
        cache_link = tmp_path / "cache2" / "beastlabai" / "multi-llm" / "ff" / "skills" \
            / "multi-llm"
        cache_link.parent.mkdir(parents=True)
        try:
            os.symlink(cache, link, target_is_directory=True)
            os.symlink(installed, cache_link, target_is_directory=True)
        except (OSError, NotImplementedError):  # pragma: no cover - no symlink right
            pytest.skip("cannot create symlinks here")
        assert _resolve("claude-code", {}, skill_root=link) == ("skill", [])
        assert _resolve("claude-code", {}, skill_root=cache_link) == (None, [])

    def test_the_default_skill_root_is_cli_commons(self, monkeypatch):
        from utils import host_capabilities

        # The capability data is read from the skill dir too; serve the real
        # document for the two fake roots so only the route decision moves.
        document = host_capabilities.load()
        skill_root = Path("/srv/p/.claude/skills/multi-llm")
        for root in (skill_root, Path(CACHE_ROOT)):
            monkeypatch.setitem(host_capabilities._CACHE, str(root), document)
        monkeypatch.setattr(cli_common, "_SKILL_DIR", skill_root)
        assert _resolve("claude-code", {}) == ("skill", [])
        monkeypatch.setattr(cli_common, "_SKILL_DIR", Path(CACHE_ROOT))
        assert _resolve("claude-code", {}) == (None, [])

    def test_the_default_environ_is_os_environ(self, monkeypatch):
        monkeypatch.setenv(modes.ROUTE_ENV, modes.TOOL_ROUTES[0])
        assert np_.resolve_route("codex") == modes.TOOL_ROUTES[0]
        monkeypatch.delenv(modes.ROUTE_ENV)
        assert np_.resolve_route("codex") is None

    def test_every_resolved_route_has_a_template(self):
        """What `resolve_route` returns, the accessor accepts (no silent fallback
        is ever needed between them)."""
        from utils import host_capabilities

        for host, env, root in [
            ("claude-code", {}, "/p/.claude/skills/multi-llm"),
            ("claude-code", {}, CACHE_ROOT),
            ("codex", {modes.ROUTE_ENV: modes.TOOL_ROUTES[0]}, None),
            ("codex", {}, None),
            ("opencode", {modes.ROUTE_ENV: modes.TOOL_ROUTES[0]}, None),
            ("opencode", {}, None),
        ]:
            route, _lines = _resolve(host, env, skill_root=root)
            assert host_capabilities.next_command_template(host, route).endswith("{args}")


@pytest.mark.parametrize("finished", ["review-code", "apply-code-fixes"])
def test_the_notice_and_the_next_step_agree_after_an_unfinished_implement(
    tmp_path, finished
):
    """#3/#4: the result's `suggested next` (workflow view) and `user_next_step`
    come from two walks; a started implement that never completed is the one
    place they used to name different phases (or `complete`)."""
    plan = _build(tmp_path, S(
        UP_TO_TASKS + ("code-review", "apply-code-fixes"),
        tasks={"T001": "completed", "T002": "pending"}, files=CR_FINDING,
    ))
    workflow = _workflow(plan)
    assert workflow["suggested_next"] == "implement"
    answer = np_.next_phase_commands(
        P, "claude-code", None, finished, plan_path=plan, workflow=workflow
    )
    assert not answer.complete
    assert answer.phase == workflow["suggested_next"]
    assert answer.command.endswith("--implement plans/foo.md --resume")


# --- resume of the SAME phase after a needs-human stop (#5, 2026-10-02) -------

class TestResumeCommand:
    """`resume_command` / `util next-command --blocked`: the one command a user
    can type after a blocked write phase, in the route's own spelling — and
    only for a phase with progress on record (adversarial review, 2026-10-02:
    every write-phase exit 3 got one, a prerequisite refusal and a
    non-existent plan included)."""

    @staticmethod
    def _in_progress(*phases):
        return {"phases": {phase: {"status": "in_progress"} for phase in phases}}

    @pytest.mark.parametrize("host,route,prefix", ROUTES, ids=ROUTE_IDS)
    def test_every_route_spells_it_for_the_user(self, host, route, prefix):
        assert np_.resume_command(
            P, host, route, "implement", workflow=self._in_progress("implement")
        ) == f"{prefix}--implement {P} --resume"

    @pytest.mark.parametrize("mode", [
        "implement", "apply-suggestions", "apply-task-suggestions", "apply-code-fixes",
    ])
    def test_the_four_write_phases_resume(self, mode):
        phase = modes.resolve(mode).phase
        assert np_.resume_command(
            P, "codex", "package", mode, workflow=self._in_progress(phase)
        ) == f"$multi-llm:multi-llm --{mode} {P} --resume"

    @pytest.mark.parametrize("mode", [
        "review-plan", "review-tasks", "review-code", "generate-tasks", "full",
        "status", "ask", "init",
    ])
    def test_every_other_mode_has_none(self, mode):
        workflow = self._in_progress(*modes.PREREQUISITE_PHASES)
        assert np_.resume_command(P, "codex", "package", mode, workflow=workflow) is None

    def test_the_plan_stays_as_typed_and_is_quoted_for_the_reader(self):
        assert np_.resume_command(
            "plans/my plan.md", "opencode", "package", "implement",
            workflow=self._in_progress("implement"),
        ) == '/multi-llm --implement "plans/my plan.md" --resume'

    def test_it_is_never_a_dispatcher_command(self):
        """Both servers redact `uv run … multi_llm.py` strings; a host spelling survives."""
        for host, route, _prefix in ROUTES:
            command = np_.resume_command(
                P, host, route, "implement", workflow=self._in_progress("implement"))
            assert "uv run" not in command and "multi_llm.py" not in command

    def test_a_route_without_a_template_raises(self):
        with pytest.raises(ValueError):
            np_.resume_command(P, "claude-code", "package", "implement",
                               workflow=self._in_progress("implement"))

    # -- which exit 3 gets one: read from state.json, the util's decision -----

    def test_a_partly_implemented_plan_resumes(self, tmp_path):
        plan = _build(tmp_path, S(UP_TO_TASKS, tasks=IMPLEMENT_PARTIAL))
        assert np_.resume_command(P, "codex", "package", "implement", plan_path=plan) \
            == f"$multi-llm:multi-llm --implement {P} --resume"

    def test_a_finalize_that_failed_resumes(self, tmp_path):
        """Every task completed, phase unmarked (`driver.verify_command` failed):
        `--resume` dispatches nothing and re-runs the check it owes."""
        plan = _build(tmp_path, S(UP_TO_TASKS, tasks={"T001": "completed", "T002": "completed"}))
        assert np_.resume_command(P, "codex", "package", "implement", plan_path=plan) \
            is not None

    def test_a_prerequisite_refusal_has_none(self, tmp_path):
        """`implement` refused for PREREQUISITE_UNMET (generate-tasks not run):
        nothing started, and `--resume` would hit the same gate."""
        plan = _build(tmp_path, S(("review-plan", "apply-suggestions")))
        assert np_.resume_command(P, "codex", "package", "implement", plan_path=plan) is None

    def test_a_stop_before_the_first_dispatch_has_none(self, tmp_path):
        """Tasks seeded, none completed (pending human-task decisions,
        `write_batch_scheduler`'s fail-closed stop): the phase's own message
        names `--human-task <id>=done|skip`, and `--resume` would stop again."""
        plan = _build(tmp_path, S(UP_TO_TASKS, tasks={"T001": "pending", "T002": "pending"}))
        assert np_.resume_command(P, "codex", "package", "implement", plan_path=plan) is None

    def test_an_apply_phase_with_a_completed_batch_resumes_and_without_one_does_not(
        self, tmp_path
    ):
        plan = _build(tmp_path, S(("review-plan",), files=RP_VALID,
                                  tasks={"apply-batch-0": "completed"}))
        assert np_.resume_command(P, "opencode", None, "apply-suggestions", plan_path=plan) \
            == f"/multi-llm --apply-suggestions {P} --resume"
        plan = _build(tmp_path / "other", S(("review-plan",), files=RP_VALID,
                                            tasks={"apply-batch-0": "failed"}))
        assert np_.resume_command(P, "opencode", None, "apply-suggestions", plan_path=plan) \
            is None

    def _run(self, argv, monkeypatch, capsys):
        record = {}
        monkeypatch.setattr(cli_common, "record_user_resume",
                            lambda obj: record.setdefault("v", obj))
        code = np_.main(argv)
        return code, capsys.readouterr(), record.get("v")

    def test_util_prints_the_line_and_sets_the_key(self, tmp_path, monkeypatch, capsys):
        _build(tmp_path, S(UP_TO_TASKS, tasks=IMPLEMENT_PARTIAL))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(modes.ROUTE_ENV, "package")
        code, out, key = self._run(
            [P, "--blocked", "implement", "--host", "codex"], monkeypatch, capsys)
        assert code == 0
        assert out.out == f"[RESUME_COMMAND] $multi-llm:multi-llm --implement {P} --resume\n"
        assert key == {"phase": "implement",
                       "command": f"$multi-llm:multi-llm --implement {P} --resume"}

    def test_util_prints_nothing_for_a_mode_with_no_resume(self, tmp_path, monkeypatch, capsys):
        _build(tmp_path, S(UP_TO_TASKS, tasks=IMPLEMENT_PARTIAL))
        monkeypatch.chdir(tmp_path)
        code, out, key = self._run(
            [P, "--blocked", "review-code", "--host", "codex"], monkeypatch, capsys)
        assert (code, out.out, key) == (0, "", None)

    def test_util_prints_nothing_for_a_phase_never_started(self, tmp_path, monkeypatch, capsys):
        """The 0defcbd symptom: `--blocked implement` after a PREREQUISITE_UNMET
        refusal printed `--implement <plan> --resume`."""
        _build(tmp_path, S(("review-plan", "apply-suggestions")))
        monkeypatch.chdir(tmp_path)
        code, out, key = self._run(
            [P, "--blocked", "implement", "--host", "codex"], monkeypatch, capsys)
        assert (code, out.out, key) == (0, "", None)

    def test_util_refuses_a_missing_plan_as_after_does(self, tmp_path, monkeypatch, capsys):
        """S5: `--blocked` skipped the plan-exists check `--after` makes;
        `util next-command does-not-exist.md --blocked implement` printed a
        resume command for a plan that does not exist."""
        monkeypatch.chdir(tmp_path)
        code, out, key = self._run(
            ["plans/ghost.md", "--blocked", "implement", "--host", "opencode"],
            monkeypatch, capsys)
        assert code == 3 and out.out == "" and key is None
        assert "plan file not found" in out.err
        assert not (tmp_path / "plans").exists()

    def test_util_reads_no_state_for_a_plan_with_none(self, tmp_path, monkeypatch, capsys):
        """A plan whose state.json does not exist: nothing is in progress, and
        the lookup creates no output dir (as `emit_answer` promises)."""
        plan = tmp_path / "plans" / "foo.md"
        plan.parent.mkdir()
        plan.write_text("# foo\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        code, out, key = self._run(
            [P, "--blocked", "implement", "--host", "opencode"], monkeypatch, capsys)
        assert (code, out.out, key) == (0, "", None)
        assert not output_dir_path(plan).exists()

    # -- generate-tasks: a failed dispatch resumes with the plain re-invoke ----
    #
    # Racing re-run 2026-10-04 (opencode #3): a generate-tasks whose one task
    # came back schema_invalid ended on a tool result with no resume_command,
    # and the host model invented `multi_llm --mode review-tasks --plan …`.
    # Its documented recovery is the plain re-invoke
    # (`run_status._RESUME_VERB_BY_MODE`), which re-dispatches the failed task.

    @staticmethod
    def _gt_status(counts, **extra):
        status = {"state": "subagents_failed", "progress": {"subagents": {
            "kind": "task_generation", "total": 1, "ok": 0, "counts": dict(counts),
            "failed_task_ids": ["task_generation"],
        }}}
        status.update(extra)
        return status

    @pytest.mark.parametrize("host,route,prefix", ROUTES, ids=ROUTE_IDS)
    def test_a_failed_generate_tasks_re_invokes_plain(self, host, route, prefix):
        assert np_.resume_command(
            P, host, route, "generate-tasks",
            phase_status=self._gt_status({"schema_invalid": 1}),
        ) == f"{prefix}--generate-tasks {P}"

    @pytest.mark.parametrize("status", [
        "timeout", "rate_limited", "runner_error", "unparseable_result",
        "schema_invalid", "incomplete",
    ])
    def test_every_rerunnable_failure_gets_it(self, status):
        assert np_.resume_command(
            P, "opencode", "package", "generate-tasks",
            phase_status=self._gt_status({status: 1}),
        ) == f"/multi-llm --generate-tasks {P}"

    @pytest.mark.parametrize("status", [
        "rejected", "invalid_model", "auth_failed", "spawn_failed",
        "sandbox_unavailable", "context_length_exceeded", "budget_exhausted",
    ])
    def test_no_command_where_no_re_run_can_help(self, status):
        """Codex #1 (same run): the CLI refused the schema before any request
        (`rejected`, rung 7, "never retried"); a re-run fails identically.

        `budget_exhausted` is the same kind of stop (fix round 6): §7.5 rule 4
        refused the slot before anything was spawned, because the configured
        timeout cannot hold one declared worst-case attempt. Its own reason says
        "Raise the slot timeout"; an unchanged re-run is refused identically."""
        assert np_.resume_command(
            P, "codex", "package", "generate-tasks",
            phase_status=self._gt_status({status: 1}),
        ) is None
        assert np_.resume_command(
            P, "codex", "package", "generate-tasks",
            phase_status=self._gt_status({"schema_invalid": 1, status: 1}),
        ) is None

    @pytest.mark.parametrize("phase_status", [
        {},
        {"state": "complete"},
        {"state": "task_generation_pending"},
        {"state": "subagents_failed"},  # no mirror: nothing says what failed
        {"state": "runner_rejected", "reason_code": "nested_exec_denied"},
    ], ids=["absent", "complete", "pending", "no-mirror", "refused"])
    def test_anything_but_a_failed_dispatch_gets_none(self, phase_status):
        assert np_.resume_command(
            P, "codex", "package", "generate-tasks", phase_status=phase_status,
        ) is None

    def test_a_refused_runner_close_out_gets_none(self):
        status = self._gt_status({"schema_invalid": 1},
                                 run={"state": "runner_rejected", "reason_code": "x"})
        assert np_.resume_command(
            P, "codex", "package", "generate-tasks", phase_status=status) is None

    def test_the_plain_re_invoke_phases_are_status_lines_plain_ones(self):
        """Held to `run_status._RESUME_VERB_BY_MODE`: the modes whose status
        line resumes with the bare `--yes` are exactly the ones answered here
        without `--resume`."""
        from utils import run_status

        plain = {mode for mode, verb in run_status._RESUME_VERB_BY_MODE.items()
                 if verb == "--yes"}
        assert {modes.resolve(m).phase for m in plain} == set(np_.RERUN_RESUME_PHASES)

    def test_util_answers_it_from_the_phase_dir(self, tmp_path, monkeypatch, capsys):
        plan = _build(tmp_path, S(("review-plan", "apply-suggestions")))
        phase_dir = output_dir_path(plan) / "generate-tasks"
        phase_dir.mkdir(parents=True)
        (phase_dir / ".status.json").write_text(
            json.dumps(self._gt_status({"schema_invalid": 1})), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(modes.ROUTE_ENV, "package")
        code, out, key = self._run(
            [P, "--blocked", "generate-tasks", "--host", "opencode"], monkeypatch, capsys)
        assert code == 0
        assert out.out == f"[RESUME_COMMAND] /multi-llm --generate-tasks {P}\n"
        assert key == {"phase": "generate-tasks",
                       "command": f"/multi-llm --generate-tasks {P}"}

    def test_after_and_blocked_exclude_each_other(self, monkeypatch, capsys):
        code, out, _key = self._run(
            [P, "--after", "implement", "--blocked", "implement"], monkeypatch, capsys)
        assert code == 3 and "mutually exclusive" in out.err

    def test_the_result_line_carries_user_resume_even_on_exit_3(self):
        ctx = cli_common.ResultContext("implement", "implement")
        ctx.set_user_resume({"phase": "implement", "command": "x"})
        assert ctx.payload(cli_common.EXIT_NEEDS_HUMAN)["user_resume"] == \
            {"phase": "implement", "command": "x"}
        assert "user_resume" not in cli_common.ResultContext(
            "implement", "implement").payload(cli_common.EXIT_NEEDS_HUMAN)
