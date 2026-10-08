"""A recorded finding ends its own sentence once on the records surfaces
(fix round 28, live run 12 #7, the summary.md and `status` part).

* Codex accept 02: summary.md's T002 row read "… in `src/config.json`.. No
  file…": the executor's CHECK FINDINGS reason put its own stop after a
  finding that ends in one.
* Claude Code stop: `status`'s why under `--human-task T002=done` ended
  "… review-code (… but the check requires 3.))": the why closed with the
  finding in brackets, and status bracketed every why again (the T005=skip
  why beside it read "… depend on it (T006))").

The findings now follow the accept why after a semicolon, without their
closing stop, and `status` prints each why under its command unbracketed,
as the ready section prints it above its command.
"""

from __future__ import annotations

import json

import pytest

from tests.test_check_gate_fixture import CHECK_GATE, DENY, PLAN, _policy  # noqa: E402
from tests.test_implement_stop_account import make_repo  # noqa: E402
from tests.test_write_executor import FakeState, env  # noqa: E402,F401 - the fixture

import multi_llm  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils import run_status  # noqa: E402
from utils import subagent_executor as executor  # noqa: E402
from utils import write_batch_scheduler as scheduler  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402

#: Claude Code stop fix 01's T002 finding, word for word.
CC_FINDING = (
    "node scripts/check.mjs exited 1 and printed 'check failed: retries must "
    "be 3 (found 2)'. src/config.json has retries=2 (set by T001) but the "
    "check requires 3."
)
#: Codex accept 02's T002 finding.
CODEX_FINDING = (
    "`node scripts/check.mjs` exited 1: retries must be 3 (found 2), in "
    "`src/config.json`."
)
ACCEPT_WHY = (
    "accept T002's check findings: T007 then run, told that you accepted the "
    "findings, and the findings stay listed for review-code; T002 reported: "
)


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def repo(tmp_path, monkeypatch):
    obj = make_repo(tmp_path, monkeypatch, PLAN)
    _policy(obj, DENY)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
        [f"T00{n}" for n in range(1, 8)]))
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
    yield obj
    registry._config = None
    registry._config_key = None


def _native_stop(repo, capsys) -> str:
    """The Claude Code stop: T001, T003, T004 done, T002's check recorded
    with the run's finding, T005 refused by the emit; `util next-command`."""
    import implement_orchestrator

    assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
    capsys.readouterr()
    phase = str(repo.phase_dir)
    for task_id in ("T001", "T003", "T004"):
        assert _util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                     task_id, "--status", "completed")[0] == 0
    assert _util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
                 "--status", "completed", "--check", "findings",
                 "--finding", CC_FINDING)[0] == 0
    state_file = str(get_or_create_state(repo.plan).state_file)
    assert _util(capsys, "finalize-tracking", "--state-file", state_file)[0] == 0
    code, out, err = _util(capsys, "next-command", "plans/checked.md",
                           "--after", "implement")
    assert code == 0, err
    return out


class TestTheAcceptWhy:

    def test_status_prints_each_why_under_its_command_with_no_bracket_of_its_own(
        self, repo, capsys,
    ):
        _native_stop(repo, capsys)
        for flags in ((), ("--view", "run")):
            run_status.main(["--plan-file", str(repo.plan), *flags])
            lines = capsys.readouterr().out.splitlines()
            at = next(i for i, line in enumerate(lines)
                      if line.endswith("--resume --human-task T002=done"))
            assert lines[at + 1] == (
                "        " + ACCEPT_WHY + CC_FINDING[:-1]), lines[at + 1]
            skip = next(i for i, line in enumerate(lines)
                        if line.endswith("--resume --human-task T005=skip"))
            assert lines[skip + 1] == (
                "        leave T005 undone, with the tasks that depend on it (T006)")
            whys = [line for line in lines if line.startswith("        ")
                    and not line.startswith("         ")]
            assert len(whys) == 7, lines
            assert not any(line.endswith(("))", ".)", ")).")) for line in whys), whys

    def test_the_ready_section_ends_the_why_without_a_stop_in_brackets(self, repo, capsys):
        out = _native_stop(repo, capsys)
        section = out.split("[NEXT_STEP_MARKDOWN]", 1)[1].split("[NEXT_STEP_MARKDOWN_END]")[0]
        assert ACCEPT_WHY + CC_FINDING[:-1] + "\n" in section, section
        assert "3.)" not in section

    def test_several_findings_are_joined_with_one_mark_between_them(self):
        answers = scheduler.compose_stop_answers(
            refused=[], failed=[], checks=["T002"], blocked={"T007": "T002"},
            findings={"T002": ["retries must be 3 (found 2).", "port is unset."]},
            undone_with=lambda _task: [],
        )
        why = next(a["why"] for a in answers if a["flags"][-1] == "T002=done")
        assert why == ACCEPT_WHY + "retries must be 3 (found 2); port is unset", why


class TestTheCheckFindingsReason:
    """The executor's reason for a check task that reported problems, which
    summary.md's row quotes (`(earlier run: check-findings: CHECK FINDINGS:
    …)` on a later run)."""

    def _task(self, env):
        return env.apply_task(
            0, kind="implement", task_id="T002", expected_ids=["T002"],
            output_name="implement_batch_T002.json",
            declared_files=[], snapshot_files=[],
        )

    def _run(self, env, findings):
        env.edits([], result={
            "status": "no-op", "ids": ["T002"], "files_touched": [],
            "summary": "the check failed", "check": {"passed": False, "findings": findings},
        })
        state = FakeState(env.tmp_path / "state.json")
        summary = env.out_dir / "implement" / "implementation_summary.md"
        result = env.scheduler(
            [self._task(env)], kind="implement", state=state,
            summary_file=str(summary),
        ).run()
        outcome = result.outcomes[0]
        assert outcome.outcome_label == executor.WRITE_LABEL_CHECK_FINDINGS, outcome.reason
        return outcome, summary.read_text(encoding="utf-8")

    def test_a_finding_that_ends_its_sentence_gets_no_second_stop(self, env):
        outcome, summary = self._run(env, [CODEX_FINDING])
        assert (f"reported 1 problem(s): {CODEX_FINDING} No file changed, so the "
                f"loop continues") in outcome.reason, outcome.reason
        assert ".." not in outcome.reason
        row = next(line for line in summary.splitlines() if line.startswith("| T002 |"))
        assert "`src/config.json`. No file" in row and ".." not in row, row

    def test_several_findings_read_one_mark_apart_and_one_stop_at_the_end(self, env):
        outcome, _summary = self._run(env, ["retries must be 3 (found 2).", "port is unset"])
        assert ("reported 2 problem(s): retries must be 3 (found 2); port is unset. "
                "No file changed") in outcome.reason, outcome.reason
