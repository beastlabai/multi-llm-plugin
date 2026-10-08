"""A code task recorded without the check its subagent reported is named
(live run 10, Claude Code racing 06).

All ten ``util batch-done`` calls of implement 06 lacked ``--own-check
passed`` although every subagent's reply said ``"check": {"passed": true}``
(``npm test`` ran each time); the same host passed it in apply-code-fixes 08.
The rule sat two paragraphs below the command block the host copies,
behind the long check-task paragraph, and ``batch-done`` said nothing.

Now the command block itself carries the check result (``{check_flags}``,
explained right under it), and ``batch-done`` answers a ``completed`` record
of a task that edits files, given no ``--own-check``, with one
``[OWN_CHECK_NOT_RECORDED]`` line naming the same call with the flag added
(and a check task given no ``--check`` with ``[CHECK_NOT_RECORDED]``), so the
host can record it in the same turn. Non-blocking and permissive: the record
is written, the exit is 0, and nothing is invented — no check result is
recorded that the host did not pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tests.test_check_findings_hold_dependents import repo  # noqa: F401 - fixture

import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from utils import batch_done  # noqa: E402
from utils.state_manager import TASK_BATCH_CHECK_KEY, get_or_create_state  # noqa: E402
from utils.subagent_tasks import PRIOR_CHANGES_TOKEN  # noqa: E402
from utils.write_phase_bridge import (  # noqa: E402
    materialize_write_tasks,
    write_task_file_for_phase,
)

SKILL_DIR = Path(__file__).resolve().parent.parent
FAILING = "npm test: 1 failing"


def _emit(repo, capsys):
    assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
    capsys.readouterr()


def _record(capsys, phase_dir, task_id, *extra, status="completed"):
    code = multi_llm.main([
        "util", "batch-done", "--phase-dir", str(phase_dir), "--task-id", task_id,
        "--status", status, *extra,
    ])
    captured = capsys.readouterr()
    assert code == 0, (captured.out, captured.err)
    return captured.out


def _marker_lines(out: str, marker: str):
    return [line for line in out.splitlines() if line.startswith(marker)]


class TestACodeTaskOfImplement:

    def test_a_completed_record_without_own_check_is_named_with_the_call(
        self, repo, capsys,
    ):
        _emit(repo, capsys)
        out = _record(capsys, repo.phase_dir, "T001")
        assert "[BATCH_DONE] T001 completed" in out
        (line,) = _marker_lines(out, batch_done.OWN_CHECK_NOT_RECORDED_MARKER)
        assert line.startswith("[OWN_CHECK_NOT_RECORDED] T001 ")
        assert (
            f'util batch-done --phase-dir "{repo.phase_dir.resolve()}" '
            f"--task-id T001 --status completed --own-check passed"
        ) in line, line
        assert "--own-check findings" in line and "--finding" in line
        assert "no `check`" in line

    def test_nothing_is_invented(self, repo, capsys):
        _emit(repo, capsys)
        _record(capsys, repo.phase_dir, "T001")
        state = get_or_create_state(repo.plan).state
        assert state["task_status"]["T001"] == "completed"
        assert TASK_BATCH_CHECK_KEY not in state
        assert "task_check_findings" not in state

    def test_recording_it_again_with_the_check_says_nothing_more(self, repo, capsys):
        _emit(repo, capsys)
        _record(capsys, repo.phase_dir, "T001")
        out = _record(capsys, repo.phase_dir, "T001", "--own-check", "passed")
        assert batch_done.OWN_CHECK_NOT_RECORDED_MARKER not in out
        assert get_or_create_state(repo.plan).state["task_status"]["T001"] == "completed"
        out = _record(capsys, repo.phase_dir, "T003", "--own-check", "findings",
                      "--finding", FAILING)
        assert "[BATCH_CHECK_FAILED] T003" in out
        assert batch_done.OWN_CHECK_NOT_RECORDED_MARKER not in out

    def test_a_failed_task_owes_no_check_result(self, repo, capsys):
        _emit(repo, capsys)
        out = _record(capsys, repo.phase_dir, "T001", "--reason", "crashed",
                      status="failed")
        assert "NOT_RECORDED]" not in out


class TestACheckTask:

    def test_a_completed_check_without_its_verdict_is_named(self, repo, capsys):
        _emit(repo, capsys)
        _record(capsys, repo.phase_dir, "T001", "--own-check", "passed")
        out = _record(capsys, repo.phase_dir, "T002")
        assert batch_done.OWN_CHECK_NOT_RECORDED_MARKER not in out
        (line,) = _marker_lines(out, batch_done.CHECK_NOT_RECORDED_MARKER)
        assert line.startswith("[CHECK_NOT_RECORDED] T002 ")
        assert "--task-id T002 --status completed --check passed" in line, line
        assert "--check findings" in line and "--check could-not-run" in line
        assert "task_check_findings" not in get_or_create_state(repo.plan).state

    def test_with_its_verdict_it_says_nothing(self, repo, capsys):
        _emit(repo, capsys)
        _record(capsys, repo.phase_dir, "T001", "--own-check", "passed")
        out = _record(capsys, repo.phase_dir, "T002", "--check", "passed")
        assert "NOT_RECORDED]" not in out


class TestAnApplyCodeFixesBatch:

    @staticmethod
    def _phase(tmp_path, monkeypatch):
        import subprocess

        from utils import provider_registry as registry

        root = tmp_path / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        config = tmp_path / "providers.yaml"
        config.write_text("driver:\n  subagent_runner: codex:gpt-5.5\n", encoding="utf-8")
        monkeypatch.setenv("MULTI_LLM_PROVIDERS_CONFIG", str(config))
        monkeypatch.setattr(registry, "_config", None)
        monkeypatch.setattr(registry, "_config_key", None)
        plan = root / "plans" / "racing.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# Racing\n", encoding="utf-8")
        out = root / "plans" / "racing"
        phase_dir = out / "apply-fixes"
        phase_dir.mkdir(parents=True)
        tasks = materialize_write_tasks(
            kind="apply_code_fixes",
            batches=[{"file_key": "index.html",
                      "prompt": f"Fix it.\n\n{PRIOR_CHANGES_TOKEN}\n",
                      "fixes": [{"group_id": "ga", "title": "Fix A"},
                                {"group_id": "gb", "title": "Fix B"}]}],
            phase_dir=str(phase_dir), plan_file=str(plan),
        )
        write_task_file_for_phase(
            kind="apply_code_fixes", tasks=tasks, phase_dir=str(phase_dir),
            plan_file=str(plan),
        )
        (out / "state.json").write_text(json.dumps({"plan_path": str(plan)}),
                                        encoding="utf-8")
        return phase_dir, tasks[0]

    def test_the_call_keeps_every_flag_the_record_was_given(
        self, tmp_path, monkeypatch, capsys,
    ):
        """A re-record without its `--declined` would record the declined fix
        applied, so the line names the call with the same flags."""
        phase_dir, task = self._phase(tmp_path, monkeypatch)
        declined = f"{task['expected_ids'][1]}=the plan forbids it"
        out = _record(capsys, phase_dir, "apply-fix-batch-0",
                      "--declined", declined, "--listed-check", "npm test=passed")
        (line,) = _marker_lines(out, batch_done.OWN_CHECK_NOT_RECORDED_MARKER)
        assert "--task-id apply-fix-batch-0 --status completed" in line
        assert "--declined '" + declined + "'" in line, line
        assert "--listed-check 'npm test=passed'" in line, line
        assert line.index("--listed-check") < line.index("--own-check passed")

        out = _record(capsys, phase_dir, "apply-fix-batch-0", "--declined", declined,
                      "--listed-check", "npm test=passed", "--own-check", "passed")
        assert batch_done.OWN_CHECK_NOT_RECORDED_MARKER not in out
        processed = json.loads((phase_dir.parent / "state.json").read_text())[
            "processed_apply-fixes"]
        assert processed[task["expected_ids"][1]]["status"] == "declined"

    def test_a_listed_failure_is_never_answered_with_a_pass(
        self, tmp_path, monkeypatch, capsys,
    ):
        """Review of fix round 24 (records #3): recorded with
        `--listed-check 'npm test=failed'` and no `--own-check`, the same call
        printed `[BATCH_CHECK_FAILED]` (the failure IS on record) and then said
        the checks its subagent ran were not on record, suggesting a re-record
        with `--own-check passed` beside the failure."""
        phase_dir, _task = self._phase(tmp_path, monkeypatch)
        out = _record(capsys, phase_dir, "apply-fix-batch-0",
                      "--listed-check", "npm test=failed")
        assert _marker_lines(out, "[BATCH_CHECK_FAILED]")
        (line,) = _marker_lines(out, batch_done.OWN_CHECK_NOT_RECORDED_MARKER)
        assert "--own-check passed" not in line, line
        assert "--own-check findings" in line and "--finding" in line
        # What is on record is said as such, and only the rest is asked for.
        assert "the checks its subagent ran on its own edits are not on record" not in line
        assert "npm test failed" in line, line
        assert '"findings"' in line, line

    def test_listed_passes_ask_only_for_the_rest_of_its_check(
        self, tmp_path, monkeypatch, capsys,
    ):
        phase_dir, _task = self._phase(tmp_path, monkeypatch)
        out = _record(capsys, phase_dir, "apply-fix-batch-0",
                      "--listed-check", "npm test=passed")
        (line,) = _marker_lines(out, batch_done.OWN_CHECK_NOT_RECORDED_MARKER)
        assert "the checks its subagent ran on its own edits are not on record" not in line
        assert "npm test passed" in line, line
        assert "--own-check passed" in line and "--own-check findings" in line


# -- the runbooks ------------------------------------------------------------


def _record_step(core: str, heading: str) -> str:
    text = (SKILL_DIR / "instructions" / core).read_text(encoding="utf-8")
    start = text.index(heading)
    return text[start:start + 2500]


def test_implement_s_command_block_carries_the_check_result():
    step = _record_step("implement.md", "**Record the task outcome**")
    block = re.search(r"```bash\n(.*?)```", step, re.S).group(1)
    assert "util batch-done" in block
    assert "--status {batch_status} {check_flags}" in block, block
    after = step.split("```", 2)[2]
    first = after.strip().split("\n", 1)[0]
    # Right under the block, before anything else: what to put there.
    assert first.startswith("`{check_flags}` is"), first
    for word in ("--own-check passed", "--own-check findings", "--check passed",
                 "[OWN_CHECK_NOT_RECORDED]"):
        assert word in first, word


def test_apply_code_fixes_command_block_carries_the_check_result():
    step = _record_step("apply-code-fixes.md", "6. **Record the batch outcome")
    block = re.search(r"```bash\n(.*?)```", step, re.S).group(1)
    assert "--status {batch_status} {check_flags}" in block, block
    after = step.split("```", 2)[2]
    first = after.strip().split("\n", 1)[0]
    assert first.startswith("`{check_flags}` is"), first
    for word in ("--own-check passed", "--own-check findings", "--listed-check",
                 "[OWN_CHECK_NOT_RECORDED]"):
        assert word in first, word
