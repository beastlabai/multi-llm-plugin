"""What a native implement (Claude Code) says about itself, fix round 16.

Live run 6 (2026-10-06), Claude Code racing 06: T011 was a manual task,
deferred under ``--yes``'s ``skip-dependents``. Implement's own result and
``implement/summary.md`` said only "T011 … deferred, left for the user"; how
to record it done (``--human-task T011=done``) first appeared in phase 08's
``[WORKFLOW_COMPLETE]`` block, two phases later. The tool routes carry that
how-to-close note (``manual_tasks_note`` / ``[MANUAL_TASK_NOTE]``) in
implement's own result. Now the native leg does too:

* the emit prints ``[MANUAL_TASK_NOTE]`` under its "Deferred tasks" list and
  hands it over as the task JSON's ``manual_tasks_note``, for the host's
  summary and report;
* ``util next-command --after implement``, the report step's lookup, prints
  the deferred tasks and the note beside the next command.

The note's command answers only the manual tasks themselves: a code task
skipped with one is not answerable ``done`` (the parser refuses a code task
that did not fail), and it runs on the resume once its manual task is done.

The fixture is ``test_human_task_answers``' plan with a code task (T006)
after the manual one (T005).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from utils import cli_common, host_capabilities  # noqa: E402
from utils import next_phase as np_  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402

from tests.test_human_task_answers import PLAN_WITH_CODE_DEPENDENT  # noqa: E402
from tests.test_implement_stop_account import make_repo  # noqa: E402

PLAN_ARG = "plans/checked.md"
MANUAL_TITLE = "Play through by hand"
DEPENDENT_TITLE = "Edit: after the manual step"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for name in ("MULTI_LLM_HOST", "MULTI_LLM_ROUTE"):
        monkeypatch.delenv(name, raising=False)
    obj = make_repo(tmp_path, monkeypatch, PLAN_WITH_CODE_DEPENDENT)
    yield obj
    registry._config = None
    registry._config_key = None


def _close_note(plan_arg: str, ids) -> str:
    """The note for *ids*, its command spelled the way Claude Code types it."""
    template = host_capabilities.next_command_template(
        "claude-code", np_.resolve_route("claude-code", stderr=io.StringIO()))
    return np_.manual_close_note(np_.close_command(template, plan_arg, ids))


def _emit(capsys, *flags: str, yes: bool = True):
    assert multi_llm.main(
        ["implement", PLAN_ARG, "--host", "claude-code",
         *(["--yes"] if yes else ["--no-confirm"]), *flags]
    ) == 0
    out = capsys.readouterr().out
    path = next(
        line.split(": ", 1)[1] for line in out.splitlines()
        if line.startswith("Output file: ")
    )
    return json.loads(Path(path).read_text(encoding="utf-8")), out


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _finish_the_native_run(repo, capsys, handoff) -> None:
    """What the host does: record each handed-out task, close the phase."""
    phase = str(repo.phase_dir)
    for batch in handoff["batches"]:
        for task in batch["tasks"]:
            extra = ["--check", "passed"] if task["id"] == "T002" else []
            code, _out, err = _util(
                capsys, "batch-done", "--phase-dir", phase, "--task-id", task["id"],
                "--status", "completed", *extra)
            assert code == 0, err
    code, _out, err = _util(
        capsys, "finalize-tracking", "--state-file", handoff["state_file"],
        "--mark-completed")
    assert code == 0, err


class TestTheEmitCarriesTheNote:

    def test_the_deferred_list_ends_with_the_note(self, repo, capsys):
        handoff, out = _emit(capsys)
        assert sorted(handoff["skipped_tasks"]) == ["T005", "T006"]
        note = _close_note(PLAN_ARG, ["T005"])
        assert handoff["manual_tasks_note"] == note
        lines = out.splitlines()
        heading = next(i for i, line in enumerate(lines)
                       if line.startswith("Deferred tasks ("))
        assert (
            "section, with the text of the [MANUAL_TASK_NOTE] line below the "
            "list under it, verbatim; give that text in your report too):"
        ) in lines[heading]
        assert f"[MANUAL_TASK_NOTE] {note}" in lines[heading:]
        # The dependent is not answerable `done`: it runs on that resume.
        assert "--human-task T005=done" in note
        assert "T006=done" not in note

    def test_the_note_is_the_command_the_resume_accepts(self, repo, capsys):
        handoff, _out = _emit(capsys)
        _finish_the_native_run(repo, capsys, handoff)
        assert multi_llm.main(
            ["implement", PLAN_ARG, "--host", "claude-code", "--yes", "--resume",
             "--human-task", "T005=done"]
        ) == 0
        capsys.readouterr()
        statuses = get_or_create_state(repo.plan).implementation_task_statuses()
        assert statuses["T005"] == "completed"

    def test_no_deferred_manual_task_no_note(self, repo, capsys):
        handoff, out = _emit(capsys, "--force-strategy", "pause-and-ask", yes=False)
        assert "manual_tasks_note" not in handoff
        assert "[MANUAL_TASK_NOTE]" not in out


class TestTheReportLookupCarriesIt:

    def test_util_next_command_after_implement_names_them_and_the_note(
        self, repo, capsys,
    ):
        handoff, _out = _emit(capsys)
        _finish_the_native_run(repo, capsys, handoff)
        code, out, err = _util(
            capsys, "next-command", PLAN_ARG, "--after", "implement",
            "--host", "claude-code")
        assert code == 0, err
        note = _close_note(PLAN_ARG, ["T005"])
        lines = [line for line in out.splitlines() if line.startswith("[")]
        assert lines[0].startswith("[NEXT_COMMAND] ")
        assert lines[0].endswith(f"--review-code {PLAN_ARG}")
        assert lines[-3:] == [
            f"[MANUAL_TASK] T005 {MANUAL_TITLE}",
            f"[MANUAL_TASK] T006 {DEPENDENT_TITLE}",
            f"[MANUAL_TASK_NOTE] {note}",
        ]
        step = cli_common.parse_result_line(out)["user_next_step"]
        assert step["complete"] is False
        assert step["command"] == lines[0].split(" ", 1)[1]
        assert step["manual_tasks"] == [
            {"id": "T005", "title": MANUAL_TITLE},
            {"id": "T006", "title": DEPENDENT_TITLE},
        ]
        assert step["manual_tasks_note"] == note

    def test_after_another_phase_it_adds_nothing(self, repo, capsys):
        handoff, _out = _emit(capsys)
        _finish_the_native_run(repo, capsys, handoff)
        code, out, _err = _util(
            capsys, "next-command", PLAN_ARG, "--after", "generate-tasks",
            "--host", "claude-code")
        assert code == 0
        assert "[MANUAL_TASK" not in out
        assert "manual_tasks" not in cli_common.parse_result_line(out)["user_next_step"]

    def test_the_complete_answer_closes_only_the_manual_task(self, repo, capsys):
        """The same rule on `[WORKFLOW_COMPLETE]`, both legs' builder: T006,
        skipped with T005, is listed as owed but never answered `done`."""
        handoff, _out = _emit(capsys)
        _finish_the_native_run(repo, capsys, handoff)
        state = get_or_create_state(repo.plan)
        state.mark_phase_completed("code-review")
        state.mark_phase_completed("apply-code-fixes")
        state.save()
        code, out, err = _util(
            capsys, "next-command", PLAN_ARG, "--after", "apply-code-fixes",
            "--host", "claude-code")
        assert code == 0, err
        assert "[WORKFLOW_COMPLETE]" in out
        assert f"[MANUAL_TASK_NOTE] {_close_note(PLAN_ARG, ['T005'])}" in out


class TestTheExecutorSummaryCarriesItToo:
    """Parity: the `--exec-subagents` leg's own summary.md listed T005 under
    `## Manual Tasks (Deferred)` with no word on how to close it either."""

    def test_the_deferred_table_ends_with_the_note(self, repo, capsys):
        repo.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
            ["T001", "T002", "T003", "T004", "T006"]))
        assert multi_llm.main(
            ["implement", PLAN_ARG, "--host", "claude-code", "--yes",
             "--exec-subagents"]
        ) == 0
        capsys.readouterr()
        report = json.loads((repo.phase_dir / "write_batch_report.json").read_text())
        summary = Path(report["summary_file"]).read_text(encoding="utf-8")
        section = summary.split("## Manual Tasks (Deferred)", 1)[1].split("\n## ", 1)[0]
        assert "| T005 |" in section and "| T006 |" in section
        assert _close_note(PLAN_ARG, ["T005"]) in section


class TestTheHostIsToldToCarryIt:

    def test_the_core_and_the_shim_say_where_the_note_goes(self):
        core = (SKILL_DIR / "instructions" / "implement.md").read_text(encoding="utf-8")
        assert core.count(
            "ending with the text of step 2's `[MANUAL_TASK_NOTE]` line (the task "
            "JSON's `manual_tasks_note`), verbatim: it says how to record them done"
        ) == 2
        assert (
            "Each manual task step 2 deferred, as the user's to do, with the "
            "`[MANUAL_TASK_NOTE]` text verbatim"
        ) in core
        skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        assert (
            "After implement those lines can also follow a `[NEXT_COMMAND]` "
            "block: list those tasks and give that note the same way, after the "
            "commands."
        ) in skill
        reference = " ".join(
            (SKILL_DIR / "references" / "summary-format.md").read_text(
                encoding="utf-8").split())
        assert (
            "A `user_next_step` that names a `command` can carry `manual_tasks` "
            "and `manual_tasks_note` too"
        ) in reference
