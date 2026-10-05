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
        plan = _complete_plan(tmp_path)
        self._persist(plan, "skip-dependents")
        assert StateManager(plan).deferred_implementation_tasks() == MANUAL

    def test_a_task_the_user_declined_is_not_a_manual_task(self, tmp_path):
        plan = _complete_plan(tmp_path)
        self._persist(plan, "pause-and-ask", {"T011": "skip"})
        assert StateManager(plan).deferred_implementation_tasks() == []

    def test_the_native_legs_reasons_read_the_same(self, tmp_path):
        """The reasons `instructions/implement.md` tells the host to record."""
        text = (SKILL_ROOT / "instructions" / "implement.md").read_text(encoding="utf-8")
        declined = "skipped by user during pause-and-ask"
        deferred = "deferred — user chose to complete manually after implementation"
        assert f'reason "{declined}"' in text and f'reason "{deferred}"' in text
        assert '"skipped — depends on skipped task {parent_id} ({parent_title})"' in text
        dependent = f"skipped — depends on skipped task T011 ({TITLES['T011']})"
        plan = _complete_plan(tmp_path)
        for parent_reason, owed in ((deferred, MANUAL), (declined, [])):
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

    def test_a_workflow_not_yet_complete_carries_no_key(self, tmp_path):
        """The tasks are named where the plugin has nothing left to run; a
        command answer keeps its shape."""
        here = os.getcwd()
        os.chdir(SKILL_ROOT)
        try:
            plan = _build(tmp_path, S(ALL_REQUIRED[:-2], tasks=DEFERRED))
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
        assert workflow["phases"]["implement"]["manual_tasks"] == [
            {"id": "T002", "title": "Implement the core"},
            {"id": "T003", "title": "Add tests"},
        ]


def test_the_tool_routes_are_told_to_list_them():
    """OpenCode's and Codex's package routes read no summary-format.md on a
    finished run: the tool description's `user_next_step` line is all they
    get before the result, so it names `manual_tasks` itself."""
    import modes

    line = modes.TOOL_USER_NEXT_STEP_LINE
    assert "manual_tasks" in line
    assert "Workflow complete." in line
