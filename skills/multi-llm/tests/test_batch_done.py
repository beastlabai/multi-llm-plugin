"""``util batch-done``: the native write leg's per-batch progress record.

Reviewer finding 1 of fix round 5 (2026-10-05): the native leg recorded a
write phase *started* at emit, so ``util next-command`` offered
``--apply-suggestions <plan> --resume`` — but nothing on that leg recorded
per-batch progress (``StateManager.mark_item_processed`` had no callers, the
instructions never wrote ``task_status``), so the offered ``--resume`` printed
``[resume] 0 already processed`` and re-emitted every batch with the same
``prior_changes_id``: a host following it re-applied batches to an
already-edited plan. The reviewer's repro (``scratchpad/review5/native-leg/
reproB``) is ``TestNativeResumeDropsRecordedBatches`` below.

The record is one command the host runs per batch, right after it verified
the batch and before it appends the batch's summary to the prior-changes log.
It writes the same ``state.json`` keys the executor leg's scheduler writes —
``task_status[<task id>]`` — plus, for the apply kinds, the base's
``processed_<phase>`` map keyed by the batch's member ids (``expected_ids``),
which is what the apply base's ``--resume`` filter reads. A native
``--resume`` whose prior-changes log names a batch it is about to emit again,
with no record saying that batch finished, is refused (exit 3) rather than
re-emitting everything.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
import apply_suggestions_orchestrator  # noqa: E402
from utils import batch_done  # noqa: E402
from utils import prior_changes  # noqa: E402
from utils.cli_common import parse_result_line  # noqa: E402
from utils.state_manager import StateManager  # noqa: E402
from utils.subagent_tasks import PRIOR_CHANGES_TOKEN  # noqa: E402
from utils.write_phase_bridge import (  # noqa: E402
    WRITE_TASK_FILENAME,
    materialize_write_tasks,
    write_task_file_for_phase,
)

from tests.test_apply_suggestions_integration import TestFixtures  # noqa: E402


INSTRUCTIONS = Path(__file__).parent.parent / "instructions"

PROMPT = "# Apply\n\n" + PRIOR_CHANGES_TOKEN + "\n\n1. Do the thing.\n"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plan_dir(tmp_path):
    plan = tmp_path / "plans" / "feature.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Feature\n\n## Step 2\n\nBody.\n", encoding="utf-8")
    out = tmp_path / "plans" / "feature"
    out.mkdir()
    return plan, out


def _suggestion_batches():
    return [
        {
            "section_key": "step_2",
            "prompt": PROMPT,
            "suggestions": [
                {"group_id": "g-email", "title": "Add email validation"},
                {"group_id": "g-password", "title": "Enforce password rules"},
            ],
        },
        {
            "section_key": "step_3",
            "prompt": PROMPT,
            "suggestions": [{"group_id": "g-rate", "title": "Add rate limiting"}],
        },
    ]


def _implement_batches():
    return [
        {"batch_index": 0, "can_parallelize": False, "tasks": [
            {"id": "T001", "title": "Foundation", "prompt": PROMPT,
             "files_to_create": ["src/a.py"], "depends_on": [],
             "executor_role": "agent"},
        ]},
        {"batch_index": 1, "can_parallelize": False, "tasks": [
            {"id": "T002", "title": "Manual step", "prompt": PROMPT,
             "files_to_create": [], "depends_on": ["T001"],
             "executor_role": "human"},
        ]},
    ]


def _materialize(plan, out, kind, batches, phase_subdir):
    phase_dir = out / phase_subdir
    phase_dir.mkdir(parents=True, exist_ok=True)
    tasks = materialize_write_tasks(
        kind=kind, batches=batches, phase_dir=str(phase_dir), plan_file=str(plan),
    )
    write_task_file_for_phase(
        kind=kind, tasks=tasks, phase_dir=str(phase_dir), plan_file=str(plan),
    )
    return phase_dir, tasks


def _run(argv, capsys):
    code = multi_llm.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err, parse_result_line(captured.out)


def _state(out):
    return json.loads((out / "state.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The util
# ---------------------------------------------------------------------------

class TestTheUtilRecordsABatch:
    def test_an_apply_batch_records_its_task_and_its_items(self, plan_dir, capsys):
        plan, out = plan_dir
        StateManager(plan).save()
        phase_dir, tasks = _materialize(
            plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        code, stdout, err, result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-batch-0", "--status", "completed",
        ], capsys)
        assert code == 0, err
        assert f"{batch_done.MARKER} apply-batch-0 completed" in stdout
        assert result["mode"] == "util batch-done"
        state = _state(out)
        assert state["task_status"]["apply-batch-0"] == "completed"
        processed = state["processed_apply-suggestions"]
        assert set(processed) == {"g-email", "g-password"}
        for item in processed.values():
            assert item["status"] == "applied"
            assert item["details"]["task_id"] == "apply-batch-0"
            assert item["details"]["prior_changes_id"] == tasks[0]["prior_changes_id"]
        # The other batch is untouched.
        assert "apply-batch-1" not in state["task_status"]
        assert "g-rate" not in processed
        # The record is readable through the StateManager API the base uses.
        reloaded = StateManager(plan, read_only=True)
        assert reloaded.is_item_processed("apply-suggestions", "g-email")
        assert reloaded.get_task_status("apply-batch-0") == "completed"

    def test_a_failed_apply_batch_records_failed_on_both(self, plan_dir, capsys):
        plan, out = plan_dir
        StateManager(plan).save()
        phase_dir, _tasks = _materialize(
            plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        code, _out, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-batch-1", "--status", "failed",
            "--reason", "subagent reported failure, plan unchanged",
        ], capsys)
        assert code == 0, err
        state = _state(out)
        assert state["task_status"]["apply-batch-1"] == "failed"
        assert state["task_status_reasons"]["apply-batch-1"] == (
            "subagent reported failure, plan unchanged")
        assert state["processed_apply-suggestions"]["g-rate"]["status"] == "failed"

    def test_apply_code_fixes_records_under_the_phase_dir_word(self, plan_dir, capsys):
        """The base keys `processed_<phase_name>` on its PHASE DIR word
        (`apply-fixes`), not the mode word the task file carries."""
        plan, out = plan_dir
        StateManager(plan).save()
        batches = [{
            "file_key": "src/a.py", "prompt": PROMPT, "batch_type": "general",
            "fixes": [{"group_id": "f-1", "title": "Null check", "file": "src/a.py"}],
        }]
        phase_dir, _tasks = _materialize(plan, out, "apply_code_fixes", batches, "apply-fixes")
        code, _out, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-fix-batch-0", "--status", "completed",
        ], capsys)
        assert code == 0, err
        state = _state(out)
        assert state["task_status"]["apply-fix-batch-0"] == "completed"
        assert "f-1" in state["processed_apply-fixes"]
        assert "processed_apply-code-fixes" not in state

    def test_an_implement_task_records_its_status_alone(self, plan_dir, capsys):
        plan, out = plan_dir
        StateManager(plan).save()
        phase_dir, _tasks = _materialize(
            plan, out, "implement", _implement_batches(), "implement")
        code, _out, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "T001", "--status", "completed",
        ], capsys)
        assert code == 0, err
        state = _state(out)
        assert state["task_status"]["T001"] == "completed"
        assert not [k for k in state if k.startswith("processed_")]

    def test_an_implement_batch_records_the_files_it_changed(
        self, plan_dir, capsys, monkeypatch
    ):
        """2026-10-05, Codex racing run 3: a partial implement left
        `tracked_files` naming the plan and `tasks.md` an earlier phase had
        recorded, and none of the code it created. On the native leg the
        closing `util finalize-tracking` is the only other writer, and a host
        that stops part-way never reaches it, so each code batch's record
        takes the census too — the output folder and the plan left out."""
        plan, out = plan_dir
        root = plan.parent.parent
        git = ["git", "-C", str(root), "-c", "user.email=t@example.com",
               "-c", "user.name=t", "-c", "commit.gpgsign=false"]
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "plans/feature.md"], check=True)
        subprocess.run(git + ["commit", "-qm", "plan"], check=True)
        head = subprocess.run(git + ["rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        monkeypatch.chdir(root)
        state = StateManager(plan)
        state.set("pre_existing_changes", ["plans/feature/"])
        state.set("head_before_implement", head)
        state.save()
        phase_dir, _tasks = _materialize(
            plan, out, "implement", _implement_batches(), "implement")
        (root / "src").mkdir()
        (root / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
        plan.write_text(plan.read_text(encoding="utf-8") + "edited\n",
                        encoding="utf-8")

        code, _out, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "T001", "--status", "completed",
        ], capsys)

        assert code == 0, err
        assert _state(out)["tracked_files"] == [
            {"path": "src/a.py", "action": "created", "task_id": "implementation"},
        ]

    def test_it_preserves_what_the_state_file_already_holds(self, plan_dir, capsys):
        """A raw read-modify-write of the host's file: a `--mark-completed`
        later re-reads it, so nothing recorded between may be lost."""
        plan, out = plan_dir
        state = StateManager(plan)
        state.mark_phase_completed("review-plan")
        state.mark_phase_started("apply-suggestions")
        state.update_task_status("apply-batch-0", "completed")
        state.save()
        phase_dir, _tasks = _materialize(
            plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        code, _out, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-batch-1", "--status", "completed",
        ], capsys)
        assert code == 0, err
        after = _state(out)
        assert after["phases_completed"]["review-plan"]
        assert after["phases_started"]["apply-suggestions"]
        assert after["task_status"] == {
            "apply-batch-0": "completed", "apply-batch-1": "completed"}

    def test_an_unknown_task_id_is_exit_3_naming_the_known_ids(self, plan_dir, capsys):
        plan, out = plan_dir
        StateManager(plan).save()
        phase_dir, _tasks = _materialize(
            plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        code, _out, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-batch-9", "--status", "completed",
        ], capsys)
        assert code == 3
        assert "apply-batch-9" in err and "apply-batch-0, apply-batch-1" in err
        assert "apply-batch-9" not in _state(out).get("task_status", {})

    def test_a_status_outside_the_two_words_is_an_argparse_error(self, plan_dir, capsys):
        plan, out = plan_dir
        phase_dir = out / "apply-suggestions"
        phase_dir.mkdir()
        with pytest.raises(SystemExit) as exc:
            batch_done.main([
                "--phase-dir", str(phase_dir), "--task-id", "apply-batch-0",
                "--status", "done",
            ])
        assert exc.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_no_task_file_is_exit_3(self, plan_dir, capsys):
        plan, out = plan_dir
        phase_dir = out / "apply-suggestions"
        phase_dir.mkdir()
        code, _out, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-batch-0", "--status", "completed",
        ], capsys)
        assert code == 3
        assert WRITE_TASK_FILENAME in err

    def test_a_human_task_is_refused(self, plan_dir, capsys):
        plan, out = plan_dir
        StateManager(plan).save()
        phase_dir, _tasks = _materialize(
            plan, out, "implement", _implement_batches(), "implement")
        code, _out, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "T002", "--status", "completed",
        ], capsys)
        assert code == 3
        assert "human task" in err


def _check_batches():
    """T001 writes a file; T002 is a check task (declares no files)."""
    return [
        {"batch_index": 0, "can_parallelize": False, "tasks": [
            {"id": "T001", "title": "Foundation", "prompt": PROMPT,
             "files_to_create": ["src/a.py"], "depends_on": [],
             "executor_role": "agent"},
        ]},
        {"batch_index": 1, "can_parallelize": False, "tasks": [
            {"id": "T002", "title": "Run static checks", "prompt": PROMPT,
             "files_to_create": [], "files_to_modify": [], "depends_on": ["T001"],
             "executor_role": "agent"},
        ]},
    ]


class TestACheckTaskRecordsWhatItFound:
    """Racing re-run 4 (2026-10-05): a check task's problems lived only in its
    runner's notes. On the native leg the host passes its reply's structured
    `check` object to `batch-done`, which writes the same `task_check_findings`
    the executor leg writes."""

    FINDINGS = ["html-validate: index.html:1 doctype", "html-validate: index.html:105 aria-label"]

    def _record(self, plan_dir, capsys, *extra, task="T002", status="completed"):
        plan, out = plan_dir
        if not (out / "state.json").exists():
            StateManager(plan).save()
        phase_dir, _tasks = _materialize(
            plan, out, "implement", _check_batches(), "implement")
        return _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", task, "--status", status, *extra,
        ], capsys)

    def test_findings_are_recorded_beside_a_completed_status(self, plan_dir, capsys):
        code, stdout, err, _result = self._record(
            plan_dir, capsys, "--check", "findings",
            "--finding", self.FINDINGS[0], "--finding", self.FINDINGS[1],
        )
        assert code == 0, err
        assert "T002 completed (2 check finding(s) recorded)" in stdout
        state = _state(plan_dir[1])
        assert state["task_status"]["T002"] == "completed"
        assert state["task_check_findings"] == {"T002": self.FINDINGS}

    def test_a_passed_check_or_a_plain_record_clears_earlier_findings(
        self, plan_dir, capsys
    ):
        self._record(plan_dir, capsys, "--check", "findings", "--finding", "x")
        code, stdout, err, _result = self._record(plan_dir, capsys, "--check", "passed")
        assert code == 0, err
        assert "check finding" not in stdout
        assert "task_check_findings" not in _state(plan_dir[1])
        self._record(plan_dir, capsys, "--check", "findings", "--finding", "x")
        self._record(plan_dir, capsys)
        assert "task_check_findings" not in _state(plan_dir[1])

    def test_a_check_with_findings_is_never_recorded_failed(self, plan_dir, capsys):
        code, _out, err, _result = self._record(
            plan_dir, capsys, "--check", "findings", "--finding", "x", status="failed")
        assert code == 3
        assert "--status completed with --check findings" in err
        # Fix round 12: the refusal names both ways a `failed` reply goes.
        assert 'said "status": "failed" but listed findings' in err
        assert "--status failed --check could-not-run" in err
        assert "T002" not in _state(plan_dir[1]).get("task_status", {})

    def test_a_check_that_could_not_run_is_recorded_failed_never_findings(
        self, plan_dir, capsys,
    ):
        """Fix round 12: the native twin of the reply's
        `check.could_not_run: true`."""
        self._record(plan_dir, capsys, "--check", "findings", "--finding", "x")
        code, stdout, err, _result = self._record(
            plan_dir, capsys, "--check", "could-not-run", status="failed")
        assert code == 0, err
        assert "[BATCH_DONE] T002 failed" in stdout
        assert "[CHECK_BLOCKED]" not in stdout
        state = _state(plan_dir[1])
        assert state["task_status"]["T002"] == "failed"
        assert state["task_status_reasons"]["T002"].startswith(
            "its check could not run")
        assert "task_check_findings" not in state
        code, _out, err, _result = self._record(
            plan_dir, capsys, "--check", "could-not-run",
            "--reason", "npx: command not found", status="failed")
        assert code == 0, err
        assert _state(plan_dir[1])["task_status_reasons"]["T002"] == (
            "npx: command not found")

    @pytest.mark.parametrize("extra,status,expected", [
        (("--check", "could-not-run"), "completed",
         "recorded --status failed with --check could-not-run"),
        (("--check", "could-not-run", "--finding", "x"), "failed",
         "--finding is recorded only with --check findings"),
    ])
    def test_could_not_run_is_failed_and_lists_no_finding(
        self, plan_dir, capsys, extra, status, expected,
    ):
        code, _out, err, _result = self._record(
            plan_dir, capsys, *extra, status=status)
        assert code == 3
        assert expected in err
        assert "T002" not in _state(plan_dir[1]).get("task_status", {})

    def test_a_code_task_records_the_checks_it_ran_and_holds_nothing(
        self, plan_dir, capsys,
    ):
        """Fix round 14 (codex racing run 5): a task that declares files
        reports the checks it ran on its own edits in the same `check`
        object. A failing one is printed for the host to list and report —
        never a plain success — and holds nothing back."""
        failing = "node tests/race-logic.test.mjs: AssertionError: a skipped gate must be rejected"
        code, stdout, err, _result = self._record(
            plan_dir, capsys, "--check", "findings", "--finding", failing, task="T001")
        assert code == 0, err
        assert "[BATCH_DONE] T001 completed" in stdout
        assert (
            "[BATCH_CHECK_FAILED] T001: its edits were applied, but its own "
            "report says a check it ran on them fails (1 problem(s)): "
            f"\"{failing}\". The edits are kept."
        ) in stdout
        assert "## Check Findings" in stdout
        assert "never report the phase as a clean success" in stdout
        assert "[CHECK_BLOCKED]" not in stdout
        state = _state(plan_dir[1])
        assert state["task_status"]["T001"] == "completed"
        assert "task_check_findings" not in state, "a write task is never a hold"

        code, stdout, err, _result = self._record(
            plan_dir, capsys, "--check", "passed", task="T001")
        assert code == 0, err
        assert "[BATCH_CHECK_FAILED]" not in stdout

    def test_a_code_task_has_no_could_not_run(self, plan_dir, capsys):
        code, _out, err, _result = self._record(
            plan_dir, capsys, "--check", "could-not-run", task="T001", status="failed")
        assert code == 3
        assert "not a check task" in err
        assert "T001" not in _state(plan_dir[1]).get("task_status", {})

    def test_a_plan_batch_has_no_check_result(self, plan_dir, capsys):
        plan, out = plan_dir
        StateManager(plan).save()
        phase_dir, _tasks = _materialize(
            plan, out, "apply_suggestions", _suggestion_batches(), "apply-suggestions")
        code, _out, err, _result = _run([
            "util", "batch-done", "--phase-dir", str(phase_dir),
            "--task-id", "apply-batch-0", "--status", "completed", "--check", "passed",
        ], capsys)
        assert code == 3
        assert "edits the plan or the task list, not code" in err

    def test_a_finding_needs_check_findings(self, plan_dir, capsys):
        code, _out, err, _result = self._record(plan_dir, capsys, "--finding", "x")
        assert code == 3
        assert "--check findings" in err


# ---------------------------------------------------------------------------
# Fix round 13: a check whose subagent changed a file is not recorded completed
# ---------------------------------------------------------------------------

class TestACheckThatChangedAFileIsNotRecordedCompleted:
    """Review of fix round 12 (native leg): `util batch-done --status
    completed --check findings` verified nothing about the tree, so a host
    following a `failed` reply that listed findings after its subagent had
    edited a file recorded check-findings, and the edit rode along. `util
    write-prompt` now records the files git sees when it binds a check's
    prompt, and the `completed` record is refused while a file changed since
    that no other task of the phase declares."""

    @pytest.fixture
    def check_repo(self, tmp_path, monkeypatch):
        root = tmp_path
        plan = root / "plans" / "feature.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Feature\n\n## Step 2\n\nBody.\n", encoding="utf-8")
        out = root / "plans" / "feature"
        out.mkdir()
        (root / "src").mkdir()
        (root / "src" / "config.json").write_text('{"retries": 2}\n', encoding="utf-8")
        (root / "src" / "b.py").write_text("b = 1\n", encoding="utf-8")
        (root / ".gitignore").write_text("logs/\n", encoding="utf-8")
        git = ["git", "-C", str(root), "-c", "user.email=t@example.com",
               "-c", "user.name=t", "-c", "commit.gpgsign=false"]
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-qm", "base"], check=True)
        monkeypatch.chdir(root)
        StateManager(plan).save()
        batches = _check_batches()
        # A task beside the check, declaring the file it may change.
        batches[1]["tasks"].append(
            {"id": "T003", "title": "Edit b", "prompt": PROMPT,
             "files_to_modify": ["src/b.py"], "depends_on": ["T001"],
             "executor_role": "agent"})
        phase_dir, _tasks = _materialize(plan, out, "implement", batches, "implement")
        return root, out, phase_dir

    def _bind(self, phase_dir, capsys, task_id="T002"):
        code, _out, err, _result = _run([
            "util", "write-prompt", "--phase-dir", str(phase_dir), "--task-id", task_id,
        ], capsys)
        assert code == 0, err

    def _done(self, phase_dir, capsys, *extra, status="completed"):
        return _run([
            "util", "batch-done", "--phase-dir", str(phase_dir), "--task-id", "T002",
            "--status", status, *extra,
        ], capsys)

    FINDINGS = ("--check", "findings", "--finding", "retries must be 3")

    @pytest.mark.parametrize("check", [
        ("--check", "findings", "--finding", "retries must be 3"),
        ("--check", "passed"),
        (),
    ])
    def test_a_file_changed_since_the_bind_refuses_the_completed_record(
        self, check_repo, capsys, check,
    ):
        root, out, phase_dir = check_repo
        self._bind(phase_dir, capsys)
        assert (phase_dir / "prompts" / "T002.tree.json").is_file()
        (root / "src" / "config.json").write_text('{"retries": 3, "mode": 1}\n', encoding="utf-8")

        code, _stdout, err, _result = self._done(phase_dir, capsys, *check)
        assert code == 3
        assert (
            "T002 is a check task (it declares no files), and since its prompt "
            "was bound with util write-prompt these files changed" in err
        ), err
        assert "src/config.json" in err
        assert '--status failed --reason "its subagent changed src/config.json"' in err
        assert "bind its prompt again with util write-prompt" in err
        state = _state(out)
        assert "T002" not in state.get("task_status", {})
        assert "task_check_findings" not in state

        # Recorded failed, as the refusal says: accepted, and the record is used up.
        code, _stdout, err, _result = self._done(
            phase_dir, capsys, "--reason", "its subagent changed src/config.json",
            status="failed")
        assert code == 0, err
        assert _state(out)["task_status"]["T002"] == "failed"
        assert not (phase_dir / "prompts" / "T002.tree.json").exists()

    def test_nothing_changed_is_recorded_and_the_record_is_used_up(
        self, check_repo, capsys,
    ):
        root, out, phase_dir = check_repo
        self._bind(phase_dir, capsys)
        # The plugin's own trees and gitignored paths are not the check's.
        (out / "prior_changes.jsonl").write_text("{}\n", encoding="utf-8")
        (root / "logs").mkdir()
        (root / "logs" / "01-implement.jsonl").write_text("{}\n", encoding="utf-8")
        # A task running beside it changes the file it declares.
        (root / "src" / "b.py").write_text("b = 22\n", encoding="utf-8")

        code, stdout, err, _result = self._done(phase_dir, capsys, *self.FINDINGS)
        assert code == 0, err
        assert "[BATCH_DONE] T002 completed" in stdout
        assert not (phase_dir / "prompts" / "T002.tree.json").exists()
        # A later record without a new bind has nothing to compare against.
        (root / "src" / "config.json").write_text('{"retries": 3, "mode": 1}\n', encoding="utf-8")
        code, _stdout, err, _result = self._done(phase_dir, capsys, "--check", "passed")
        assert code == 0, err

    def test_binding_again_takes_a_new_record(self, check_repo, capsys):
        root, out, phase_dir = check_repo
        self._bind(phase_dir, capsys)
        (root / "src" / "config.json").write_text('{"retries": 3, "mode": 1}\n', encoding="utf-8")
        assert self._done(phase_dir, capsys, *self.FINDINGS)[0] == 3
        self._bind(phase_dir, capsys)
        code, _stdout, err, _result = self._done(phase_dir, capsys, *self.FINDINGS)
        assert code == 0, err
        assert _state(out)["task_status"]["T002"] == "completed"

    def test_a_check_bound_without_write_prompt_is_recorded_as_before(
        self, check_repo, capsys,
    ):
        root, out, phase_dir = check_repo
        (root / "src" / "config.json").write_text('{"retries": 3, "mode": 1}\n', encoding="utf-8")
        code, _stdout, err, _result = self._done(phase_dir, capsys, *self.FINDINGS)
        assert code == 0, err

    def test_a_task_that_declares_files_takes_no_record(self, check_repo, capsys):
        _root, _out, phase_dir = check_repo
        self._bind(phase_dir, capsys, task_id="T003")
        assert not (phase_dir / "prompts" / "T003.tree.json").exists()


# ---------------------------------------------------------------------------
# Fix round 8: one lock around every raw read-modify-write of state.json
# ---------------------------------------------------------------------------

SKILL_DIR = Path(__file__).parent.parent


def _parallel_batches():
    """One implement batch whose two tasks run in parallel (implement.md lets
    a host dispatch same-batch tasks together, and record each as it lands)."""
    return [{"batch_index": 0, "can_parallelize": True, "tasks": [
        {"id": "T001", "title": "Left", "prompt": PROMPT,
         "files_to_create": ["src/a.py"], "depends_on": [], "executor_role": "agent"},
        {"id": "T002", "title": "Right", "prompt": PROMPT,
         "files_to_create": ["src/b.py"], "depends_on": [], "executor_role": "agent"},
    ]}]


def _child(code: str, *args: str, cwd) -> subprocess.Popen:
    """A separate process (the host runs each util call as its own command)."""
    return subprocess.Popen(
        [sys.executable, "-c", code, str(SKILL_DIR), *map(str, args)],
        cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


#: `util batch-done` with a rendezvous inside its read-modify-write: each
#: record says it got there, then waits (bounded) for the other one.
_RENDEZVOUS_RECORD = r"""
import os, sys, time
sys.path.insert(0, sys.argv[1])
from utils import batch_done
sync_dir, phase_dir, task_id = sys.argv[2:5]
census = batch_done._tracked_after_batch
def at_the_census(*args, **kwargs):
    open(os.path.join(sync_dir, task_id), "w").close()
    deadline = time.monotonic() + 3.0
    while len(os.listdir(sync_dir)) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    return census(*args, **kwargs)
batch_done._tracked_after_batch = at_the_census
sys.exit(batch_done.main(["--phase-dir", phase_dir, "--task-id", task_id,
                          "--status", "completed"]))
"""

#: Each raw writer of state.json, as the host runs it.
_WRITERS = {
    "batch-done": (
        "from utils import batch_done as m",
        ["--phase-dir", "{phase_dir}", "--task-id", "T001", "--status", "completed"],
        lambda state: state.get("task_status", {}).get("T001") == "completed",
    ),
    "metrics record": (
        "from utils import metrics as m",
        ["record", "--state-file", "{state}", "--phase", "implement",
         "--label", "T001: Left", "--no-eta"],
        lambda state: len(state.get("metrics", {}).get("implement", [])) == 1,
    ),
    "metrics start": (
        "from utils import metrics as m",
        ["start", "--state-file", "{state}", "--phase", "apply-suggestions",
         "--total-batches", "2"],
        lambda state: "apply-suggestions" in state.get("metrics_progress", {}),
    ),
    "metrics finish": (
        "from utils import metrics as m",
        ["finish", "--state-file", "{state}", "--phase", "implement"],
        lambda state: "implement" not in state.get("metrics_progress", {}),
    ),
    "finalize-tracking": (
        "import finalize_tracking as m",
        ["--state-file", "{state}", "--mark-completed"],
        lambda state: "implement" in state.get("phases_completed", {}),
    ),
}


class TestStateWritesAreSerialized:
    """`util batch-done` read `state.json`, ran the dirty-tree census (git
    status, ls-files, diff: seconds on a real repository since fix round 7)
    and wrote the whole file back, unlocked. implement.md lets same-batch
    tasks run in parallel and a host issues the records in parallel too, so
    two records read the same file and the second write dropped the first
    one's `task_status` (fix round 8, reviewer finding 3; measured: two
    records started together left only `T001`). Every raw read-modify-write
    of `state.json` now holds one lock, `<state dir>/.state.lock`."""

    @pytest.fixture
    def implement_phase(self, plan_dir):
        plan, out = plan_dir
        state = StateManager(plan)
        state.mark_phase_started("implement")
        state.save()
        phase_dir, _tasks = _materialize(
            plan, out, "implement", _parallel_batches(), "implement")
        return out / "state.json", phase_dir

    def test_two_parallel_records_keep_both_task_statuses(
        self, implement_phase, tmp_path
    ):
        state_path, phase_dir = implement_phase
        sync = tmp_path / "sync"
        sync.mkdir()

        children = [
            _child(_RENDEZVOUS_RECORD, sync, phase_dir, task, cwd=tmp_path)
            for task in ("T001", "T002")
        ]
        for child in children:
            _out, err = child.communicate(timeout=60)
            assert child.returncode == 0, err

        assert json.loads(state_path.read_text(encoding="utf-8"))["task_status"] == {
            "T001": "completed", "T002": "completed",
        }

    @pytest.mark.parametrize("writer", sorted(_WRITERS))
    def test_every_raw_state_writer_waits_for_the_lock(
        self, writer, implement_phase, tmp_path
    ):
        from utils import metrics

        state_path, phase_dir = implement_phase
        state = json.loads(state_path.read_text(encoding="utf-8"))
        metrics.start_phase_in_state(state, "implement", 2)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        load, argv, wrote = _WRITERS[writer]
        argv = [a.format(state=state_path, phase_dir=phase_dir) for a in argv]
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); " + load
            + "; sys.exit(m.main(sys.argv[2:]))"
        )

        with metrics.state_file_lock(state_path):
            child = _child(code, *argv, cwd=tmp_path)
            try:
                child.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                pass
            assert child.poll() is None, (
                f"{writer} wrote state.json while the lock was held: "
                f"{child.communicate()}"
            )
            held = json.loads(state_path.read_text(encoding="utf-8"))
            held["written_while_locked"] = True
            state_path.write_text(json.dumps(held), encoding="utf-8")

        _out, err = child.communicate(timeout=60)
        assert child.returncode == 0, err
        after = json.loads(state_path.read_text(encoding="utf-8"))
        assert after.get("written_while_locked") is True
        assert wrote(after)

    def test_a_lock_never_released_fails_the_record_and_writes_nothing(
        self, implement_phase, tmp_path, monkeypatch, capsys
    ):
        from utils import metrics

        state_path, phase_dir = implement_phase
        before = state_path.read_text(encoding="utf-8")
        holder = _child(
            "import sys, time; sys.path.insert(0, sys.argv[1]); "
            "from utils import metrics; "
            "lock = metrics.state_file_lock(sys.argv[2]); lock.__enter__(); "
            "print('held', flush=True); time.sleep(30)",
            state_path, cwd=tmp_path,
        )
        try:
            assert holder.stdout.readline().strip() == "held"
            monkeypatch.setattr(metrics, "STATE_LOCK_TIMEOUT_S", 0.3)

            code = batch_done.main([
                "--phase-dir", str(phase_dir), "--task-id", "T001",
                "--status", "completed",
            ])
        finally:
            holder.kill()
            holder.communicate()

        assert code == 1
        err = capsys.readouterr().err
        assert ".state.lock" in err and "nothing was written" in err
        assert state_path.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# The reviewer's repro: a real apply-suggestions run, then --resume
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path, monkeypatch):
    """`reproB/build.py`: a git repo holding the integration fixture's plan,
    its review-plan artifacts, and review-plan recorded completed."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    plan = root / "plans" / "test-plan.md"
    review = root / "plans" / "test-plan" / "review-plan"
    review.mkdir(parents=True)
    plan.write_text(TestFixtures.PLAN_CONTENT, encoding="utf-8")
    (review / "grouped.json").write_text(
        json.dumps(TestFixtures.GROUPED_SUGGESTIONS, indent=2), encoding="utf-8")
    (review / "validation.json").write_text(
        json.dumps(TestFixtures.VALIDATION_RESULTS, indent=2), encoding="utf-8")
    (review / "backup.md").write_text(TestFixtures.PLAN_CONTENT, encoding="utf-8")
    state = StateManager(plan)
    state.mark_phase_completed("review-plan")
    state.save()
    monkeypatch.chdir(root)
    return plan


def _emit(plan, capsys, *flags):
    """One native apply-suggestions run; its exit, stderr and the task file."""
    code = apply_suggestions_orchestrator.main([
        "--plan-file", str(plan), "--yes", "--output-format", "json", *flags,
    ])
    captured = capsys.readouterr()
    task_file = plan.parent / "test-plan" / "apply-suggestions" / WRITE_TASK_FILENAME
    tasks = (json.loads(task_file.read_text(encoding="utf-8"))["batches"]
             if task_file.exists() else [])
    return code, captured.err, tasks


def _batch_done(plan, task_id, status="completed"):
    return batch_done.main([
        "--phase-dir", str(plan.parent / "test-plan" / "apply-suggestions"),
        "--task-id", task_id, "--status", status,
    ])


class TestNativeResumeDropsRecordedBatches:
    def test_a_recorded_batch_is_not_emitted_again(self, repo, capsys):
        plan = repo
        code, err, first = _emit(plan, capsys)
        assert code == 0, err
        assert len(first) == 3, [t["task_id"] for t in first]
        done = first[0]
        # The host's per-batch steps, in the instructions' order: record,
        # then append the summary to the log.
        assert _batch_done(plan, done["task_id"]) == 0
        capsys.readouterr()
        assert prior_changes.append(
            str(plan.parent / "test-plan"), done["prior_changes_id"],
            "apply-suggestions", "batch 0 applied")

        code, err, second = _emit(plan, capsys, "--resume")
        assert code == 0, err
        assert f"[resume] {len(done['expected_ids'])} already processed" in err
        # The recorded batch's items are gone from the re-emitted set, and
        # the batch that applied them is never re-dispatched under its id.
        emitted_items = {i for t in second for i in t["expected_ids"]}
        assert emitted_items.isdisjoint(done["expected_ids"])
        assert done["prior_changes_id"] not in {t["prior_changes_id"] for t in second}
        assert emitted_items == {
            i for t in first[1:] for i in t["expected_ids"]}

    def test_a_failed_batch_is_retried(self, repo, capsys):
        """`--status failed` is a record too — the batch is dispatched again
        on resume, as the executor leg re-dispatches a failed batch, and its
        log entry (`(batch failed — …)`) earns no refusal."""
        plan = repo
        code, err, first = _emit(plan, capsys)
        assert code == 0, err
        failed = first[1]
        assert _batch_done(plan, failed["task_id"], "failed") == 0
        capsys.readouterr()
        prior_changes.append(
            str(plan.parent / "test-plan"), failed["prior_changes_id"],
            "apply-suggestions", "(batch failed — timeout, verified no file changes)")
        code, err, second = _emit(plan, capsys, "--resume")
        assert code == 0, err
        assert set(failed["expected_ids"]) <= {
            i for t in second for i in t["expected_ids"]}

    def test_a_started_phase_with_a_batch_recorded_reads_in_progress(self, repo, capsys):
        """The record is what `status` and the next-step builder already
        read (`apply-batch-N` completed), so the offered `--resume` is now
        backed by a record the resume honours."""
        import check_workflow_prerequisites as prereq
        from utils import next_phase

        plan = repo
        code, err, first = _emit(plan, capsys)
        assert code == 0, err
        assert _batch_done(plan, first[0]["task_id"]) == 0
        capsys.readouterr()
        status = prereq.get_workflow_status(
            StateManager(plan, read_only=True), plan, plan.parent / "test-plan")
        assert status["phases"]["apply-suggestions"]["status"] == "in_progress"
        answer = next_phase.next_phase_commands(
            "plans/test-plan.md", "claude-code", None, None, plan_path=plan)
        assert answer.command.endswith("--apply-suggestions plans/test-plan.md --resume")


class TestNativeResumeRefusesUnrecordedLoggedBatches:
    def test_a_logged_batch_without_a_record_refuses_the_resume(self, repo, capsys):
        """reproB as the reviewer ran it: one batch appended to the log and
        nothing recorded. Re-emitting every batch would re-apply it; the
        resume refuses and names the batch and the record command."""
        plan = repo
        code, err, first = _emit(plan, capsys)
        assert code == 0, err
        logged = first[0]
        prior_changes.append(
            str(plan.parent / "test-plan"), logged["prior_changes_id"],
            "apply-suggestions", "batch 0 applied")
        code, err, _second = _emit(plan, capsys, "--resume")
        assert code == 3
        assert batch_done.RESUME_REFUSED_MARKER in err
        assert logged["task_id"] in err and logged["prior_changes_id"] in err
        assert "util batch-done" in err and "--fresh" in err
        # Nothing re-stamped: the phase is still the first run's start.
        assert "apply-suggestions" in _state(plan.parent / "test-plan")["phases_started"]

        # The repair the message names, then the resume goes through.
        assert _batch_done(plan, logged["task_id"]) == 0
        capsys.readouterr()
        code, err, second = _emit(plan, capsys, "--resume")
        assert code == 0, err
        assert logged["prior_changes_id"] not in {t["prior_changes_id"] for t in second}

    def test_a_log_entry_of_another_phase_is_not_this_phases(self, repo, capsys):
        plan = repo
        code, err, first = _emit(plan, capsys)
        assert code == 0, err
        prior_changes.append(
            str(plan.parent / "test-plan"), first[0]["prior_changes_id"],
            "implement", "some implement task")
        code, err, _second = _emit(plan, capsys, "--resume")
        assert code == 0, err

    def test_a_bare_rerun_is_not_refused(self, repo, capsys):
        """Only `--resume` promises to skip landed batches; a bare re-run
        starts the phase over, as documented, and is not this check's."""
        plan = repo
        code, err, first = _emit(plan, capsys)
        assert code == 0, err
        prior_changes.append(
            str(plan.parent / "test-plan"), first[0]["prior_changes_id"],
            "apply-suggestions", "batch 0 applied")
        code, err, _second = _emit(plan, capsys)
        assert code == 0, err


# ---------------------------------------------------------------------------
# The helper the two native emit paths share
# ---------------------------------------------------------------------------

class TestUnrecordedLoggedBatches:
    def test_implement_keys_on_task_status(self, plan_dir):
        plan, out = plan_dir
        state = StateManager(plan)
        state.save()
        _phase_dir, tasks = _materialize(
            plan, out, "implement", _implement_batches(), "implement")
        prior_changes.append(str(out), "T001", "implement", "did T001")
        unrecorded = batch_done.unrecorded_logged_batches(
            tasks, output_dir=str(out), phase="implement", state=state.state)
        assert [t["task_id"] for t in unrecorded] == ["T001"]
        for recorded_as in ("completed", "failed"):
            state.update_task_status("T001", recorded_as)
            assert batch_done.unrecorded_logged_batches(
                tasks, output_dir=str(out), phase="implement", state=state.state) == []
        state.update_task_status("T001", "pending")
        assert len(batch_done.unrecorded_logged_batches(
            tasks, output_dir=str(out), phase="implement", state=state.state)) == 1

    def test_apply_code_fixes_reads_the_aliased_phase_word(self, plan_dir):
        """The native instruction appends under `apply-fixes`, the executor
        under `apply-code-fixes`; both name one phase to the log."""
        plan, out = plan_dir
        state = StateManager(plan)
        state.save()
        batches = [{
            "file_key": "src/a.py", "prompt": PROMPT, "batch_type": "general",
            "fixes": [{"group_id": "f-1", "title": "Null check", "file": "src/a.py"}],
        }]
        _phase_dir, tasks = _materialize(plan, out, "apply_code_fixes", batches, "apply-fixes")
        prior_changes.append(str(out), tasks[0]["prior_changes_id"], "apply-fixes", "fixed")
        unrecorded = batch_done.unrecorded_logged_batches(
            tasks, output_dir=str(out), phase="apply-code-fixes", state=state.state,
            processed_phase="apply-fixes")
        assert [t["task_id"] for t in unrecorded] == ["apply-fix-batch-0"]
        state.mark_item_processed("apply-fixes", "f-1", "applied")
        assert batch_done.unrecorded_logged_batches(
            tasks, output_dir=str(out), phase="apply-code-fixes", state=state.state,
            processed_phase="apply-fixes") == []


# ---------------------------------------------------------------------------
# The instructions run it
# ---------------------------------------------------------------------------

class TestTheInstructionsRecordEachBatch:
    """The four write-mode cores record every batch through `util batch-done`
    right after verification and BEFORE the prior-changes append, so a host
    that dies between the two leaves a record the resume honours, never a log
    entry the resume would re-apply."""

    @pytest.mark.parametrize("core,phase_dir,id_example", [
        ("apply-suggestions.md", "apply-suggestions", "apply-batch-0"),
        ("apply-task-suggestions.md", "apply-task-suggestions", "apply-task-batch-0"),
        ("apply-code-fixes.md", "apply-fixes", "apply-fix-batch-0"),
        ("implement.md", "implement", "T001"),
    ])
    def test_the_loop_records_each_batch_before_it_appends(self, core, phase_dir, id_example):
        text = (INSTRUCTIONS / core).read_text(encoding="utf-8")
        record = re.search(
            r'util batch-done \\\n\s+--phase-dir "\{plan\}/' + re.escape(phase_dir)
            + r'" --task-id "[^"]+" --status \{batch_status\}', text)
        assert record, core
        append = text.index("util prior-changes append \\\n")
        assert record.start() < append, (core, "record the batch, then append")
        # The two outcomes the record takes, named for the host.
        assert "`completed`" in text and "`failed`" in text, core
        assert "[BATCH_DONE]" in text, core

    def test_implement_no_longer_asks_the_host_to_retype_a_state_write(self):
        text = (INSTRUCTIONS / "implement.md").read_text(encoding="utf-8")
        assert 'state["task_status"][task_id]' not in text

    @pytest.mark.parametrize("core", [
        "apply-suggestions.md", "apply-task-suggestions.md", "apply-code-fixes.md",
    ])
    def test_the_resume_section_names_what_the_resume_honours(self, core):
        text = (INSTRUCTIONS / core).read_text(encoding="utf-8")
        assert "util batch-done" in text.split("### Resuming Interrupted Sessions", 1)[1]
