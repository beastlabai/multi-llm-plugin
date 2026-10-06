"""A complete workflow names the manual tasks the user still owes.

Measured 2026-10-04 (OpenCode racing re-run): `implement` deferred two file-less
manual playthrough tasks (T011, T012) under `--yes`'s `skip-dependents`, and
`apply-code-fixes` then printed `[WORKFLOW_COMPLETE]` / "Workflow complete." with
no word about them. The workflow IS complete — the plugin can do nothing more —
but every completion surface now names what is left for the user:

* the builder's answer (``user_next_step.manual_tasks``) and the block it
  renders (one ``[MANUAL_TASK]`` line per task under ``[WORKFLOW_COMPLETE]``);
* the workflow payload (``phases.implement.manual_tasks``) and the status view's
  trailer line, which a terminal ``--view run`` poll carries too;
* the Stop hook's ``util report-check``, which passes a final message that says
  "Workflow complete." and names each task, and blocks one that drops them.

The deferred set is read from ``state.json`` (an implement task recorded
``skipped``), never by re-parsing tasks.md; the titles come from the
``task_titles`` map ``implement`` records when it decomposes the task list.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import check_workflow_prerequisites as prereq  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, detach, report_check, run_status, status_file  # noqa: E402
from utils import next_phase as np_  # noqa: E402
from utils.output_handler import output_dir_path  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402

from tests.test_next_phase import ALL_REQUIRED, CR_FINDING, P, S, _build  # noqa: E402
from tests.test_report_check import Transcript, decision_of, run_check  # noqa: E402
from tests.test_write_executor_orchestrators import (  # noqa: E402,F401
    IMPLEMENT_WRITE_IDS,
    _seed_generate_tasks,
    repo,
)

SKILL_ROOT = Path(__file__).parent.parent
PLUGIN = "/multi-llm:multi-llm "

TITLES = {
    "T001": "Page shell",
    "T010": "Static validation of the inline module",
    "T011": "Manual playthrough over the static server",
    "T012": "Manual file:// load and blocked-CDN fallback check",
}
DEFERRED = {"T001": "completed", "T010": "completed",
            "T011": "skipped", "T012": "skipped"}
MANUAL = [
    {"id": "T011", "title": TITLES["T011"]},
    {"id": "T012", "title": TITLES["T012"]},
]
#: The same two when T012 was skipped because it depends on T011 (fix round
#: 17: the entry says what it waits on).
WAITING = [MANUAL[0], dict(MANUAL[1], waits_on="T011")]


def close_note(prefix: str, plan: str, ids) -> str:
    """The `manual_tasks_note` for *ids*, its command spelled out by hand
    (fix round 14: how a deferred manual task is recorded done)."""
    answers = " ".join(f"--human-task {task_id}=done" for task_id in ids)
    command = f"{prefix}--implement {plan} --resume {answers}"
    assert np_.manual_close_note(command).count(command) == 1
    return np_.manual_close_note(command)


def _complete_plan(root: Path, *, titles=True, tasks=None) -> Path:
    """Every required phase on record, two implement tasks deferred."""
    here = os.getcwd()
    os.chdir(SKILL_ROOT)  # the first StateManager runs `git rev-parse HEAD`
    try:
        plan = _build(root, S(ALL_REQUIRED, tasks=tasks or DEFERRED, files=CR_FINDING))
    finally:
        os.chdir(here)
    if titles:
        state = StateManager(plan)
        state.record_task_titles(TITLES)
        state.save()
    return plan


def _workflow(plan: Path) -> dict:
    return prereq.get_workflow_status(StateManager(plan), plan, output_dir_path(plan))


# ---------------------------------------------------------------------------
# The state reader and the workflow payload
# ---------------------------------------------------------------------------

class TestTheDeferredSetComesFromState:
    def test_skipped_implement_tasks_with_their_titles(self, tmp_path):
        plan = _complete_plan(tmp_path)
        assert StateManager(plan).deferred_implementation_tasks() == MANUAL

    def test_a_task_with_no_recorded_title_is_named_by_id(self, tmp_path):
        """A state written before titles were recorded (the 2026-10-04 run's own)."""
        plan = _complete_plan(tmp_path, titles=False)
        assert StateManager(plan).deferred_implementation_tasks() == [
            {"id": "T011", "title": None}, {"id": "T012", "title": None},
        ]

    def test_another_phases_skipped_batch_is_not_an_implement_task(self, tmp_path):
        plan = _complete_plan(tmp_path, tasks={
            "T001": "completed", "apply-fix-batch-0": "skipped",
        })
        assert StateManager(plan).deferred_implementation_tasks() == []

    def test_the_workflow_payload_carries_them_on_implement(self, tmp_path):
        plan = _complete_plan(tmp_path)
        payload = _workflow(plan)
        assert payload["suggested_next"] is None
        assert payload["phases"]["implement"]["manual_tasks"] == MANUAL

    def test_nothing_deferred_adds_no_key(self, tmp_path):
        plan = _complete_plan(tmp_path, tasks={"T001": "completed"})
        assert "manual_tasks" not in _workflow(plan)["phases"]["implement"]


class TestOnlyTheCurrentDeferralsAreOwed:
    """Adversarial review 2026-10-05: `task_status` is never reset on a fresh
    implement and the titles merge forever, so after tasks.md was regenerated
    an earlier cycle's skipped `T020` was printed as `[MANUAL_TASK]` (and the
    Stop hook blocked the final message until the model named a task that no
    longer exists); and a task the user declined was labelled manual."""

    def test_a_skip_from_an_earlier_decomposition_is_not_owed(self, tmp_path):
        plan = _complete_plan(tmp_path, titles=False,
                              tasks=dict(DEFERRED, T020="skipped"))
        state = StateManager(plan)
        state.record_task_titles({"T020": "Old cycle's manual check"})
        state.record_task_titles(TITLES)  # the decomposition implement ran last
        state.save()
        assert StateManager(plan).deferred_implementation_tasks() == MANUAL

    def test_the_decomposition_is_recorded_whole_each_time(self, tmp_path):
        plan = _complete_plan(tmp_path)
        state = StateManager(plan)
        assert state.record_task_titles({"T011": TITLES["T011"]}) is True
        assert state.record_task_titles({"T011": TITLES["T011"]}) is False
        state.save()
        assert StateManager(plan).deferred_implementation_tasks() == MANUAL[:1]

    def test_the_recorded_graph_does_not_depend_on_the_task_lists_order(
        self, tmp_path,
    ):
        """Fix round 13: each task's dependency list was stored in the order
        tasks.md happened to list it, so the same graph recorded two ways
        (and the container goldens that record `implement_task_depends_on`)
        differed. Each list is sorted and de-duplicated now."""
        plan = _complete_plan(tmp_path)
        state = StateManager(plan)
        assert state.record_task_titles(
            TITLES, depends_on={
                "T012": ["T011", "T003", "T011", "", "T001"],
                "T011": ("T002", "T001"),
                "T001": None,
            },
        ) is True
        assert state.get("implement_task_depends_on") == {
            "T001": [], "T011": ["T001", "T002"], "T012": ["T001", "T003", "T011"],
        }
        # The same graph in another order is no change.
        assert state.record_task_titles(
            TITLES, depends_on={
                "T011": ["T001", "T002"], "T001": [],
                "T012": ["T001", "T011", "T003"],
            },
        ) is False

    @staticmethod
    def _persist(plan, strategy, decisions=None):
        """The executor leg's own reasons, through its own persistence."""
        from utils.write_batch_scheduler import resolve_human_task_plan

        tasks = [
            {"task_id": "T010", "executor_role": "agent", "task_title": "Code"},
            {"task_id": "T011", "executor_role": "human",
             "task_title": TITLES["T011"]},
            {"task_id": "T012", "executor_role": "agent", "depends_on": ["T011"],
             "task_title": TITLES["T012"]},
        ]
        human = resolve_human_task_plan(tasks, strategy=strategy, decisions=decisions)
        state = StateManager(plan)
        state.state["task_status"] = {"T001": "completed", "T010": "completed"}
        for task_id, reason in human.skipped.items():
            state.update_task_status(task_id, "skipped", reason=reason)
        state.save()

    def test_a_deferred_human_task_and_its_dependent_are_owed(self, tmp_path):
        """Both are owed; the dependent says what it waits on (fix round 17:
        implement runs it once T011 is recorded done)."""
        plan = _complete_plan(tmp_path)
        self._persist(plan, "skip-dependents")
        assert StateManager(plan).deferred_implementation_tasks() == WAITING

    def test_a_task_the_user_declined_is_not_a_manual_task(self, tmp_path):
        plan = _complete_plan(tmp_path)
        self._persist(plan, "pause-and-ask", {"T011": "skip"})
        assert StateManager(plan).deferred_implementation_tasks() == []

    def test_the_native_host_records_no_skip_by_hand(self):
        """Fix round 7: the orchestrator records every skip on the native leg
        too, so the instructions no longer hand the host a reason to write."""
        text = (SKILL_ROOT / "instructions" / "implement.md").read_text(encoding="utf-8")
        assert 'Mark all human tasks as "skipped" in state' not in text
        assert 'Mark task as "skipped" in state' not in text
        assert 'state["human_task_strategy"]' not in text
        assert "--force-strategy <answer>" in text
        assert "--resume --human-task {id}=skip" in text
        assert '"skipped — depends on skipped task {parent_id} ({parent_title})"' in text

    def test_an_older_native_hosts_reasons_read_the_same(self, tmp_path):
        """The reasons the instructions told the host to record before fix
        round 7, still in states written then."""
        declined = "skipped by user during pause-and-ask"
        deferred = "deferred — user chose to complete manually after implementation"
        dependent = f"skipped — depends on skipped task T011 ({TITLES['T011']})"
        plan = _complete_plan(tmp_path)
        for parent_reason, owed in ((deferred, WAITING), (declined, [])):
            state = StateManager(plan)
            state.update_task_status("T011", "skipped", reason=parent_reason)
            state.update_task_status("T012", "skipped", reason=dependent)
            state.save()
            assert StateManager(plan).deferred_implementation_tasks() == owed


# ---------------------------------------------------------------------------
# The builder and the block
# ---------------------------------------------------------------------------

class TestTheCompleteAnswerNamesThem:
    def test_user_next_step_gains_manual_tasks(self, tmp_path):
        plan = _complete_plan(tmp_path)
        answer = np_.next_phase_commands(
            P, "claude-code", None, "apply-code-fixes", plan_path=plan
        )
        assert answer.complete is True
        assert answer.to_dict() == {
            "phase": None, "command": None, "complete": True,
            "manual_tasks": MANUAL,
            "manual_tasks_note": close_note(PLUGIN, P, ["T011", "T012"]),
        }

    def test_the_fallback_walk_names_them_too(self, tmp_path):
        """`full` and `util next-command` with no `--after` answer from
        `suggested_next`: the same completion, the same tasks."""
        plan = _complete_plan(tmp_path)
        answer = np_.next_phase_commands(P, "codex", "package", "full", plan_path=plan)
        assert answer.to_dict()["manual_tasks"] == MANUAL

    def test_the_block_lists_one_line_per_task_under_the_marker(self, tmp_path):
        plan = _complete_plan(tmp_path)
        answer = np_.next_phase_commands(
            P, "claude-code", None, "apply-code-fixes", plan_path=plan
        )
        assert np_.render_next_phase_lines(answer) == [
            "[WORKFLOW_COMPLETE]",
            f"[MANUAL_TASK] T011 {TITLES['T011']}",
            f"[MANUAL_TASK] T012 {TITLES['T012']}",
            "[MANUAL_TASK_NOTE] " + close_note(PLUGIN, P, ["T011", "T012"]),
        ]

    def test_an_untitled_task_renders_its_id_alone(self):
        answer = np_.NextPhase(
            phase=None, command=None, complete=True,
            manual_tasks=(np_.ManualTask("T010", None),),
        )
        assert np_.render_next_phase_lines(answer) == [
            "[WORKFLOW_COMPLETE]", "[MANUAL_TASK] T010",
        ]
        assert answer.to_dict()["manual_tasks"] == [{"id": "T010", "title": None}]

    def test_a_command_answer_names_them_and_how_to_close_them(self, tmp_path):
        """Fix round 16 (run 6, Codex racing 06r/07): the hosts relay
        `user_next_step`, and only the complete answer of 08 carried the
        manual task's closing command — 06r and 07 named T009 from the
        notice without it. A command answer now carries both keys beside
        its command, and stays a command answer: not complete. Its block
        lists them after the command, as `[WORKFLOW_COMPLETE]` does (fix
        round 17: round 16's native package renders them there, and this
        test still said the block was the command alone)."""
        here = os.getcwd()
        os.chdir(SKILL_ROOT)
        try:
            plan = _build(tmp_path, S(ALL_REQUIRED[:-2], tasks=DEFERRED))
        finally:
            os.chdir(here)
        state = StateManager(plan)
        state.record_task_titles(TITLES)
        state.save()
        answer = np_.next_phase_commands(
            P, "claude-code", None, "implement", plan_path=plan
        )
        assert answer.complete is False
        payload = answer.to_dict()
        assert payload["complete"] is False and payload["command"]
        assert payload["manual_tasks"] == MANUAL
        assert payload["manual_tasks_note"] == close_note(PLUGIN, P, ["T011", "T012"])
        assert np_.render_next_phase_lines(answer) == [
            f"[NEXT_COMMAND] {payload['command']}",
            f"[MANUAL_TASK] T011 {TITLES['T011']}",
            f"[MANUAL_TASK] T012 {TITLES['T012']}",
            "[MANUAL_TASK_NOTE] " + close_note(PLUGIN, P, ["T011", "T012"]),
        ]

    def test_a_command_answer_with_nothing_deferred_keeps_its_shape(self, tmp_path):
        here = os.getcwd()
        os.chdir(SKILL_ROOT)
        try:
            plan = _build(tmp_path, S(ALL_REQUIRED[:-2]))
        finally:
            os.chdir(here)
        answer = np_.next_phase_commands(
            P, "claude-code", None, "implement", plan_path=plan
        )
        assert answer.complete is False
        assert "manual_tasks" not in answer.to_dict()


# ---------------------------------------------------------------------------
# The emitters
# ---------------------------------------------------------------------------

@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE", detach.PLAN_ARG_ENV,
                 detach.WORKER_ENV):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _run(argv, capsys):
    code = multi_llm.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err, cli_common.parse_result_line(captured.out)


class TestTheEmittersCarryThem:
    def test_util_next_command(self, workdir, capsys):
        _complete_plan(workdir)
        code, out, err, result = _run(
            ["util", "next-command", P, "--after", "apply-code-fixes"], capsys
        )
        assert code == 0, err
        assert result["user_next_step"] == {
            "phase": None, "command": None, "complete": True,
            "manual_tasks": MANUAL,
            "manual_tasks_note": close_note(PLUGIN, P, ["T011", "T012"]),
        }
        lines = out.splitlines()
        start = lines.index("[WORKFLOW_COMPLETE]")
        assert lines[start:start + 3] == [
            "[WORKFLOW_COMPLETE]",
            f"[MANUAL_TASK] T011 {TITLES['T011']}",
            f"[MANUAL_TASK] T012 {TITLES['T012']}",
        ]

    def test_the_status_view_names_them(self, workdir, capsys):
        plan = _complete_plan(workdir)
        run_status.main(["--plan-file", str(plan)])
        out = capsys.readouterr().out
        assert (
            f"manual tasks left for you: T011 {TITLES['T011']}; "
            f"T012 {TITLES['T012']}"
        ) in out
        # Fix round 14: and how to record them done, in the host's spelling.
        assert close_note(PLUGIN, str(plan), ["T011", "T012"]) in out

    def test_a_terminal_run_poll_carries_them_into_its_notice(self, workdir, capsys):
        """The opencode plugin reads only `--view run`: the F7 carry-over."""
        plan = _complete_plan(workdir)
        phase_dir = output_dir_path(plan) / "apply-fixes"
        phase_dir.mkdir(parents=True, exist_ok=True)
        status_file.write_status(str(phase_dir), {
            "phase": "apply-code-fixes", "state": "complete",
        })
        status_file.claim_run(str(phase_dir), {
            "run_id": "run-manual-0001", "state": "done", "exit_code": 0,
            "detached": True, "mode": "apply-code-fixes",
        })
        ctx = cli_common.activate_result(cli_common.ResultContext())
        try:
            assert run_status.main([
                "--plan-file", str(plan), "--view", "run",
                "--phase", "apply-code-fixes",
            ]) == 0
            payload = ctx.payload(0)
        finally:
            cli_common.deactivate_result()
        out = capsys.readouterr().out
        assert "workflow manual tasks left for you: T011" in out
        assert "manual tasks left for you: T011" in payload["notice"]


# ---------------------------------------------------------------------------
# The Stop hook's report check
# ---------------------------------------------------------------------------

COMPLETE_WITH_MANUAL = (
    "[WORKFLOW_COMPLETE]\n"
    f"[MANUAL_TASK] T011 {TITLES['T011']}\n"
    f"[MANUAL_TASK] T012 {TITLES['T012']}\n"
    'MULTI_LLM_RESULT: {"mode": "util next-command", "phase": null, "state": "done", '
    '"exit": 0, "artifacts": [], "next": null}\n'
)


class TestTheReportCheck:
    def test_the_block_parses_its_manual_tasks(self):
        block = report_check.find_blocks(COMPLETE_WITH_MANUAL)[-1]
        assert block.complete is True
        assert block.manual == [("T011", TITLES["T011"]), ("T012", TITLES["T012"])]

    def test_a_message_that_lists_them_passes(self, tmp_path, capsys):
        message = (
            "## Apply Code Fixes Complete\n\n### Next Steps\n\nWorkflow complete.\n\n"
            "Manual tasks still yours (deferred by implement):\n"
            f"- T011 — {TITLES['T011']}\n- T012 — {TITLES['T012']}\n"
        )
        path = (Transcript().prompt("go").bash(COMPLETE_WITH_MANUAL).say(message)
                .write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_a_message_that_drops_them_blocks_naming_them(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(COMPLETE_WITH_MANUAL)
                .say("## Done\n\n### Next Steps\n\nWorkflow complete.")
                .write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert '"Workflow complete."' in reason
        assert f"T011 {TITLES['T011']}" in reason
        assert f"T012 {TITLES['T012']}" in reason

    def test_naming_one_of_two_still_blocks(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(COMPLETE_WITH_MANUAL)
                .say("Workflow complete. Left for you: T011.")
                .write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert "T012" in reason

    def test_an_id_inside_a_longer_token_does_not_count(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(COMPLETE_WITH_MANUAL)
                .say("Workflow complete. See T0110 and xT012.")
                .write(tmp_path / "t.jsonl"))
        assert "decision" in run_check(capsys, path)


NEXT = f"{PLUGIN}--review-code plans/foo.md"
NOTE = close_note(PLUGIN, "plans/foo.md", ["T011", "T012"])
COMMAND_WITH_MANUAL = (
    f"[NEXT_COMMAND] {NEXT}\n"
    f"[MANUAL_TASK] T011 {TITLES['T011']}\n"
    f"[MANUAL_TASK] T012 {TITLES['T012']}\n"
    f"[MANUAL_TASK_NOTE] {NOTE}\n"
    'MULTI_LLM_RESULT: {"mode": "util next-command", "phase": null, "state": "done", '
    '"exit": 0, "artifacts": [], "next": null}\n'
)


class TestTheReportCheckAfterACommand:
    """Fix round 17 (review of round 16): since round 16 the same lines
    follow a `[NEXT_COMMAND]` block while implement's manual tasks are
    owed, and SKILL.md and summary-format.md ask the report to list them
    after the commands; the Stop hook still read them under
    `[WORKFLOW_COMPLETE]` only, so a report that dropped them passed. It
    checks their ids there too: the same word match, which a report that
    lists them as asked always passes."""

    def test_the_block_parses_them_and_the_note(self):
        block = report_check.find_blocks(COMMAND_WITH_MANUAL)[-1]
        assert block.complete is False and block.command == NEXT
        assert block.manual == [("T011", TITLES["T011"]), ("T012", TITLES["T012"])]
        assert block.note == NOTE

    def test_a_message_with_the_command_and_the_tasks_passes(self, tmp_path, capsys):
        message = (
            "## Review Done\n\n### Next Steps\n\n```\n" + NEXT + "\n```\n\n"
            f"Manual tasks still yours:\n- T011 {TITLES['T011']}\n"
            f"- T012 {TITLES['T012']}\n\n{NOTE}\n"
        )
        path = (Transcript().prompt("go").bash(COMMAND_WITH_MANUAL).say(message)
                .write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""

    def test_a_message_that_drops_them_blocks_naming_them(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(COMMAND_WITH_MANUAL)
                .say("### Next Steps\n\n```\n" + NEXT + "\n```\n")
                .write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert "does not name the tasks T011, T012" in reason
        # The reply replaces the message: the command, the tasks, the note.
        assert f"```\n{NEXT}\n```" in reason
        assert f"- T011 {TITLES['T011']}" in reason
        assert f"- T012 {TITLES['T012']}" in reason
        assert NOTE in reason
        assert 'never write "Workflow complete."' in reason

    def test_a_message_that_drops_the_command_too_names_both(self, tmp_path, capsys):
        path = (Transcript().prompt("go").bash(COMMAND_WITH_MANUAL)
                .say("### Next Steps\n\nRun the review next. T011 is yours.")
                .write(tmp_path / "t.jsonl"))
        reason = decision_of(run_check(capsys, path))["reason"]
        assert f"Missing command:\n\n```\n{NEXT}\n```" in reason
        assert f"- T012 {TITLES['T012']}" in reason

    def test_a_command_block_with_no_tasks_is_checked_as_before(self, tmp_path, capsys):
        bare = COMMAND_WITH_MANUAL.split("[MANUAL_TASK]", 1)[0]
        path = (Transcript().prompt("go").bash(bare)
                .say("### Next Steps\n\n```\n" + NEXT + "\n```\n")
                .write(tmp_path / "t.jsonl"))
        assert run_check(capsys, path) == ""


# ---------------------------------------------------------------------------
# implement records the titles; the report rule says what to write
# ---------------------------------------------------------------------------

def test_the_report_rule_tells_the_host_to_list_them():
    text = (SKILL_ROOT / "references" / "summary-format.md").read_text(encoding="utf-8")
    assert "manual_tasks" in text


class TestImplementRecordsTheTitles:
    """Both legs pass through `implement_orchestrator.main`'s decomposition,
    which is where the titles are recorded; the native leg's host then writes
    `task_status[T…] = "skipped"` itself and the title is already on record."""

    def test_the_native_leg_records_every_task_title(self, repo):
        import implement_orchestrator

        _seed_generate_tasks(repo)
        assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
        state = json.loads((output_dir_path(repo.plan) / "state.json").read_text())
        assert state["task_titles"] == {
            "T001": "Create the module",
            "T002": "Implement the core",
            "T003": "Add tests",
        }
        # No manual task, no key: a plan without one keeps its state shape.
        assert "implement_human_task_ids" not in state

    def test_the_executor_leg_names_its_deferred_tasks_on_completion(self, repo):
        import implement_orchestrator

        _seed_generate_tasks(repo)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        original = implement_orchestrator.TaskDecomposer

        class SeededDecomposer(original):
            def parse_plan(self, content):
                tasks = super().parse_plan(content)
                self.tasks["T002"].executor_role = "human"
                self.tasks["T002"].subagent_type = "human"
                return tasks

        repo.monkeypatch.setattr(implement_orchestrator, "TaskDecomposer", SeededDecomposer)
        assert implement_orchestrator.main(
            ["--plan-file", str(repo.plan), "--yes", "--exec-subagents"]
        ) == 0
        workflow = _workflow(repo.plan)
        assert workflow["phases"]["implement"]["manual_tasks"] == OWED_T002_T003

    def test_the_executor_leg_result_names_how_to_close_them(self, repo, capsys):
        """Run 6, Codex racing 06r: implement finished with a manual task
        deferred, and its result's `user_next_step` (what the tool routes
        relay) named review-code and nothing about recording the task done;
        the command was only in the `notice`. It carries the note now. The
        note answers the manual task alone (fix round 17: this test still
        expected `--human-task T003=done` for the code task skipped with
        T002, which implement refuses; T003 runs on that resume)."""
        import implement_orchestrator

        _seed_generate_tasks(repo)
        repo.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps([
            {"path": "src/main.py", "mode": "patch", "content": "# edited\n"}
        ]))
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", IMPLEMENT_WRITE_IDS)
        original = implement_orchestrator.TaskDecomposer

        class SeededDecomposer(original):
            def parse_plan(self, content):
                tasks = super().parse_plan(content)
                self.tasks["T002"].executor_role = "human"
                self.tasks["T002"].subagent_type = "human"
                return tasks

        repo.monkeypatch.setattr(implement_orchestrator, "TaskDecomposer", SeededDecomposer)
        capsys.readouterr()
        assert multi_llm.main([
            "implement", str(repo.plan), "--yes", "--exec-subagents", "--host", "codex",
        ]) == 0
        step = cli_common.parse_result_line(capsys.readouterr().out)["user_next_step"]
        assert step["complete"] is False
        assert "--review-code" in step["command"]
        assert [task["id"] for task in step["manual_tasks"]] == ["T002", "T003"]
        assert (
            "--implement " in step["manual_tasks_note"]
            and "--resume --human-task T002=done " in step["manual_tasks_note"]
        ), step
        assert "T003=done" not in step["manual_tasks_note"], step


def test_the_tool_routes_are_told_to_list_them():
    """OpenCode's and Codex's package routes read no summary-format.md on a
    finished run: the tool description's `user_next_step` line is all they
    get before the result, so it names `manual_tasks` itself."""
    import modes

    line = modes.TOOL_USER_NEXT_STEP_LINE
    assert "manual_tasks" in line
    assert "Workflow complete." in line


def test_the_tool_routes_are_told_to_relay_them_on_any_result():
    """Fix round 16 (run 6, Codex racing 06r/07): the hosts named the
    manual task T009 from the notice but not `--human-task T009=done`;
    only 08's complete answer carried the note. The tool description and
    every route's relay rule now say to list `manual_tasks` and give the
    note on any result, complete or not."""
    import modes

    line = modes.TOOL_USER_NEXT_STEP_LINE
    assert "whenever it carries `manual_tasks`, complete or not" in line
    assert "then give its `manual_tasks_note` verbatim" in line
    for template in ("hosts/codex/plugin-skill.md.tmpl", "hosts/codex/shim.md.tmpl",
                     "hosts/opencode/shim.md.tmpl", "hosts/opencode/plugin-command.md.tmpl"):
        text = " ".join((SKILL_ROOT / template).read_text(encoding="utf-8").split())
        assert (
            "When it carries `manual_tasks`, complete or not, also list each "
            "(id and title) and give its `manual_tasks_note` verbatim"
        ) in text, template
    rule = " ".join(
        (SKILL_ROOT / "references/summary-format.md").read_text(encoding="utf-8").split())
    assert "A `user_next_step` that names a `command` can carry `manual_tasks`" in rule


def test_no_doc_says_only_the_complete_block_renders_them():
    """Fix round 17 (review of round 16): `next_phase.py`'s field comment and
    AGENTS.md's `user_next_step` sentence said only the complete block
    renders the manual tasks, while `render_next_phase_lines` lists them
    after every command block that carries them (pinned above)."""
    for path in ("utils/next_phase.py", "AGENTS.md", "SKILL.md",
                 "references/summary-format.md"):
        text = " ".join((SKILL_ROOT / path).read_text(encoding="utf-8").split()).lower()
        assert "only the complete block renders them" not in text, path
    agents = " ".join((SKILL_ROOT / "AGENTS.md").read_text(encoding="utf-8").split())
    assert "every block renders them, after a command block's commands too" in agents


def test_the_tool_routes_say_workflow_complete_only_on_a_complete_result():
    """OpenCode racing run 4, phase 06: after implement, with a `command`
    naming review-code and a notice listing the manual tasks left, the host
    wrote "Workflow complete. Manual tasks left …". The line now says when
    the sentence belongs, and that it never does otherwise."""
    import modes

    line = modes.TOOL_USER_NEXT_STEP_LINE
    assert "when it is `complete`, and only then, say \"Workflow complete.\"" in line
    assert "Never say \"Workflow complete.\" otherwise" in line
    assert "even when a notice lists manual tasks" in line


# ---------------------------------------------------------------------------
# The native leg records its own deferrals (fix round 7)
# ---------------------------------------------------------------------------

def _human_t002(repo):
    """The repo fixture's three-task chain with T002 a manual task."""
    import implement_orchestrator

    original = implement_orchestrator.TaskDecomposer

    class SeededDecomposer(original):
        def parse_plan(self, content):
            tasks = super().parse_plan(content)
            self.tasks["T002"].executor_role = "human"
            self.tasks["T002"].subagent_type = "human"
            return tasks

    repo.monkeypatch.setattr(implement_orchestrator, "TaskDecomposer", SeededDecomposer)


def _native_emit(repo, capsys, *flags):
    """`implement` on the native leg (no `--exec-subagents`), via the dispatcher."""
    code, out, err, result = _run(
        ["implement", str(repo.plan), "--host", "claude-code", *flags], capsys
    )
    assert code == 0, err
    output_file = next(
        line.split("Output file:", 1)[1].strip()
        for line in out.splitlines() if line.startswith("Output file:")
    )
    return out, json.loads(Path(output_file).read_text(encoding="utf-8"))


def _state(repo) -> dict:
    return json.loads((output_dir_path(repo.plan) / "state.json").read_text())


def _host_runs_t001_and_closes(repo, capsys):
    """What the host does on the native leg: T001's subagent edits, its record,
    then the closing `util finalize-tracking --mark-completed`."""
    (repo.root / "src" / "main.py").write_text("x = 2\n", encoding="utf-8")
    phase_dir = output_dir_path(repo.plan) / "implement"
    state_file = output_dir_path(repo.plan) / "state.json"
    for argv in (
        ["util", "batch-done", "--phase-dir", str(phase_dir), "--task-id", "T001",
         "--status", "completed"],
        ["util", "finalize-tracking", "--state-file", str(state_file),
         "--touched", "src/main.py", "--mark-completed"],
    ):
        code, _out, err, _result = _run(argv, capsys)
        assert code == 0, err


OWED_T002_T003 = [
    {"id": "T002", "title": "Implement the core"},
    # T003 depends on the manual T002: implement runs it once T002 is
    # recorded done (fix round 17).
    {"id": "T003", "title": "Add tests", "waits_on": "T002"},
]


class TestTheNativeLegRecordsItsDeferrals:
    """Racing re-run 3 (2026-10-05, Claude Code): T010, a manual task deferred
    under `--yes`, was "Deferred" in summary.md but never reached `task_status`
    — the native leg left that write to the host, which only ever records the
    tasks it dispatches — so after apply-code-fixes `util next-command` printed
    a bare `[WORKFLOW_COMPLETE]`. The emit now resolves and records the
    human-task plan itself, from the strategy and the tasks it already holds."""

    def test_yes_records_the_deferral_and_its_dependent_at_emit(self, repo, capsys):
        _seed_generate_tasks(repo)
        _human_t002(repo)
        out, handoff = _native_emit(repo, capsys, "--yes")

        state = _state(repo)
        assert state["task_status"]["T002"] == "skipped"
        assert state["task_status_reasons"]["T002"].startswith(
            "deferred — manual task"
        )
        assert state["task_status"]["T003"] == "skipped"
        assert state["task_status_reasons"]["T003"].startswith(
            "skipped — depends on skipped task T002"
        )
        assert "T001" not in state["task_status"]  # the host's to record
        assert state["implement_human_task_ids"] == ["T002"]

        # The host reads the skips; it does not re-derive them.
        assert sorted(handoff["skipped_tasks"]) == ["T002", "T003"]
        by_id = {t["id"]: t for b in handoff["batches"] for t in b["tasks"]}
        # Fix round 14: named in `skipped_tasks`, never handed out in the batches.
        assert "T002" not in by_id and "T003" not in by_id
        assert handoff["skipped_tasks"]["T003"] == state["task_status_reasons"]["T003"]
        assert by_id["T001"]["status"] == "pending"
        assert "T002 Implement the core: deferred — manual task" in out

    def test_the_whole_native_path_names_the_manual_tasks_on_completion(
        self, repo, capsys
    ):
        """native implement under --yes → batch-done → finalize --mark-completed
        → review-code done → apply-code-fixes' closing call → next-command."""
        _seed_generate_tasks(repo)
        _human_t002(repo)
        _native_emit(repo, capsys, "--yes")
        _host_runs_t001_and_closes(repo, capsys)

        # review-code finished with one finding (its own completion write).
        review_dir = output_dir_path(repo.plan) / "code-review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "grouped.json").write_text(json.dumps({
            "format_version": 2, "groups": [{
                "group_id": "g1", "theme": "t",
                "suggestions": [{"title": "s", "importance": "HIGH"}],
            }],
        }), encoding="utf-8")
        state = StateManager(repo.plan)
        state.mark_phase_completed("review-plan")
        state.mark_phase_completed("code-review")
        state.save()

        plan_arg = "plans/sample.md"
        complete = {
            "phase": None, "command": None, "complete": True,
            "manual_tasks": OWED_T002_T003,
            # Fix round 16: T003, skipped with the manual T002, is listed as
            # owed but never answered `done` (implement refuses a code task
            # that did not fail); it runs on that resume.
            "manual_tasks_note": close_note(PLUGIN, plan_arg, ["T002"]),
        }
        block = [
            "[WORKFLOW_COMPLETE]",
            "[MANUAL_TASK] T002 Implement the core",
            "[MANUAL_TASK] T003 Add tests "
            "(waits on T002: implement runs it once T002 is recorded done)",
        ]
        code, out, err, result = _run(
            ["apply-code-fixes", plan_arg, "--mark-completed"], capsys
        )
        assert code == 0, err
        assert result["user_next_step"] == complete
        lines = out.splitlines()
        start = lines.index("[WORKFLOW_COMPLETE]")
        assert lines[start:start + 3] == block

        code, out, err, result = _run(
            ["util", "next-command", plan_arg, "--after", "apply-code-fixes"], capsys
        )
        assert code == 0, err
        assert result["user_next_step"] == complete
        lines = out.splitlines()
        start = lines.index("[WORKFLOW_COMPLETE]")
        assert lines[start:start + 3] == block

    def test_a_manual_task_nobody_answered_is_owed_once_implement_closes(
        self, repo, capsys
    ):
        """The backstop: an attended run whose host picked a strategy and
        recorded nothing. The finished phase still names the manual task —
        never a clean completion — while a running one does not."""
        _seed_generate_tasks(repo)
        _human_t002(repo)
        _out, handoff = _native_emit(repo, capsys)  # attended: no strategy yet
        assert handoff["human_task_strategy"] is None
        assert "skipped_tasks" not in handoff
        assert "T002" not in _state(repo)["task_status"]
        assert StateManager(repo.plan).deferred_implementation_tasks() == []

        _host_runs_t001_and_closes(repo, capsys)
        assert StateManager(repo.plan).deferred_implementation_tasks() == [
            {"id": "T002", "title": "Implement the core"},
        ]

    def test_a_done_answer_is_recorded_and_not_owed(self, repo, capsys):
        _seed_generate_tasks(repo)
        _human_t002(repo)
        _out, handoff = _native_emit(
            repo, capsys, "--force-strategy", "pause-and-ask",
            "--human-task", "T002=done",
        )
        assert _state(repo)["task_status"]["T002"] == "completed"
        assert "skipped_tasks" not in handoff
        _host_runs_t001_and_closes(repo, capsys)
        assert StateManager(repo.plan).deferred_implementation_tasks() == []

    def test_a_declined_task_and_its_dependent_are_recorded_and_not_owed(
        self, repo, capsys
    ):
        _seed_generate_tasks(repo)
        _human_t002(repo)
        _out, handoff = _native_emit(
            repo, capsys, "--force-strategy", "pause-and-ask",
            "--human-task", "T002=skip",
        )
        state = _state(repo)
        assert state["task_status_reasons"]["T002"] == (
            "skipped by user during pause-and-ask"
        )
        assert state["task_status"]["T003"] == "skipped"
        assert sorted(handoff["skipped_tasks"]) == ["T002", "T003"]
        _host_runs_t001_and_closes(repo, capsys)
        assert StateManager(repo.plan).deferred_implementation_tasks() == []


def test_the_claude_code_host_is_told_to_relay_the_note():
    """Fix round 15 (round 14's leftover): the close note reached the tool
    routes' `user_next_step` line and summary-format.md, but the Claude Code
    shim, where that host learns what a `[WORKFLOW_COMPLETE]` lookup means,
    and full-workflow.md's `### Next Steps` block named only the manual
    tasks, never the `[MANUAL_TASK_NOTE]` line saying how to record them
    done."""
    skill = " ".join((SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").split())
    assert (
        "A lookup that prints `[WORKFLOW_COMPLETE]` ends the report with "
        "\"Workflow complete.\", each `[MANUAL_TASK]` line's task (id and "
        "title) as the user's to do, keeping a line's \"(waits on …)\" note "
        "(implement runs that task once the one it waits on is recorded "
        "done), then the `[MANUAL_TASK_NOTE]` line's text verbatim"
    ) in skill
    core = " ".join(
        (SKILL_ROOT / "instructions" / "full-workflow.md").read_text(encoding="utf-8").split()
    )
    assert (
        "with each of its `manual_tasks` listed under it, then its "
        "`manual_tasks_note` (the `[MANUAL_TASK_NOTE]` line) verbatim"
    ) in core
