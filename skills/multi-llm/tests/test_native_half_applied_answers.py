"""A native task that failed with its edits half-applied gets the executor's answers.

Fix round 21 (review of round 20, stops #3). The executor offers
``--human-task <id>=done`` / ``=skip`` for a failed task only when it failed
cleanly, with nothing moved (``WRITE_CLEAN_FAILURE`` and not ``bytes_moved``),
and none at all for a partial application, which is inspected first: its
stop says "inspect the half-applied state, restore or keep it, fix the
cause, then resume this phase". The native leg's ``native_stop_answers``
offered ``done`` / ``skip`` for every ``failed`` task — "accept T003's
failure: T003 counts as done and the tasks after it run" for a task whose
subagent stopped half-way through its edits — because ``util batch-done
--status failed`` recorded nothing about what the task left behind.

``util batch-done --status failed --partial`` now records it (the reason's
:data:`HALF_APPLIED_PREFIX`), and the native answers withhold that task's
as the executor does; a reply that said ``applied`` and changed nothing
(the reason the instructions prescribe, ``said applied, no file changed``)
gets the executor's unlanded order, ``skip`` before ``done``.

Driven on the check-gate fixture: T001 done, T002's check reports findings
(holding T007), T003 fails after its subagent edited ``src/main.py``.
"""

from __future__ import annotations

import json

import pytest

from tests.test_check_gate_fixture import (  # noqa: E402
    CHECK_GATE,
    FINDINGS,
    PLAN,
    _check,
    _policy,
)
from tests.test_implement_stop_account import make_repo  # noqa: E402

import multi_llm  # noqa: E402
from utils import provider_registry as registry  # noqa: E402
from utils.state_manager import get_or_create_state  # noqa: E402

ALLOW = (
    "driver:\n  subagent_runner: codex:gpt-5.5\n  write_paths:\n"
    "    allow: ['**']\n    deny: ['.git/**', '.multi-llm/**']\n"
)
GREETING = "Title: Edit: add the greeting module"
#: T002's three answers, the only ones either leg offers at this stop.
CHECK_ONLY = [
    ["--resume", "--human-task", "T002=done"],
    ["--resume", "--human-task", "T002=skip"],
    ["--resume", "--human-task", "T002=retry"],
]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    obj = make_repo(tmp_path, monkeypatch, PLAN)
    _policy(obj, ALLOW)
    monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
        [f"T00{n}" for n in range(1, 8)]))
    monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
    yield obj
    registry._config = None
    registry._config_key = None


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _emit(repo, capsys) -> None:
    import implement_orchestrator

    assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
    capsys.readouterr()


def _answers(capsys) -> list:
    """The native stop's answers as `util next-command` prints them."""
    code, out, err = _util(capsys, "next-command", "plans/checked.md",
                           "--after", "implement")
    assert code == 0, err
    pairs, why = [], None
    for line in out.splitlines():
        if line.startswith("[NEXT_COMMAND_ANSWERS_WHY] "):
            why = line.split(" ", 1)[1]
        elif line.startswith("[NEXT_COMMAND_ANSWERS] "):
            pairs.append((why, line.split(" --implement plans/checked.md ", 1)[1].split()))
            why = None
    return pairs


def _native_stop(repo, capsys, *t003: str) -> None:
    """T001 done, T002's check with findings, T003 recorded failed with
    *t003* after its subagent changed src/main.py; the phase closed without
    --mark-completed."""
    _emit(repo, capsys)
    phase = str(repo.phase_dir)
    assert _util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T001",
                 "--status", "completed")[0] == 0
    assert _util(capsys, "batch-done", "--phase-dir", phase, "--task-id", "T002",
                 "--status", "completed", "--check", "findings",
                 "--finding", FINDINGS[0])[0] == 0
    main = repo.root / "src" / "main.py"
    main.write_text(main.read_text(encoding="utf-8") + "def greet(:\n", encoding="utf-8")
    code, out, err = _util(capsys, "batch-done", "--phase-dir", phase, "--task-id",
                           "T003", "--status", "failed", *t003)
    assert code == 0, err
    state_file = str(get_or_create_state(repo.plan).state_file)
    assert _util(capsys, "finalize-tracking", "--state-file", state_file)[0] == 0
    return out


class TestAHalfAppliedFailureOffersNoAnswer:

    def test_without_the_record_it_is_answered_as_a_clean_failure(self, repo, capsys):
        """What the host can still record: a failed task with a reason alone
        is a clean failure, answered as the executor answers one."""
        _native_stop(repo, capsys, "--reason", "the greeting module failed to build")
        flags = [flags for _why, flags in _answers(capsys)]
        assert ["--resume", "--human-task", "T003=done"] in flags
        assert ["--resume", "--human-task", "T003=skip"] in flags

    def test_a_half_applied_task_gets_none_of_its_own(self, repo, capsys):
        out = _native_stop(repo, capsys, "--reason", "stopped half-way", "--partial")
        assert "[BATCH_DONE] T003 failed" in out
        assert "T003's edits are half-applied" in out, out
        reason = get_or_create_state(repo.plan).state["task_status_reasons"]["T003"]
        assert reason == "half-applied: stopped half-way"
        pairs = _answers(capsys)
        assert [flags for _why, flags in pairs] == CHECK_ONLY, pairs
        assert not any("T003" in why for why, _flags in pairs)

    def test_the_reason_defaults_to_saying_so(self, repo, capsys):
        _native_stop(repo, capsys, "--partial")
        reason = get_or_create_state(repo.plan).state["task_status_reasons"]["T003"]
        assert reason.startswith("half-applied: ")

    def test_status_names_the_resume_and_no_answer_for_it(self, repo, capsys):
        from utils import run_status

        _native_stop(repo, capsys, "--partial")
        capsys.readouterr()
        run_status.main(["--plan-file", "plans/checked.md", "--view", "run",
                         "--host", "claude-code"])
        out = capsys.readouterr().out
        assert "resume with: /multi-llm:multi-llm --implement plans/checked.md --resume" in out
        assert "T003=done" not in out and "T003=skip" not in out
        assert "T002=done" in out

    def test_it_is_the_executor_stops_shape(self, repo, capsys, tmp_path, monkeypatch):
        """The same story on the --exec-subagents leg: the runner edits
        src/main.py for T003 and reports failed, a partial application. Its
        stop records the same answers, in the same order."""
        _native_stop(repo, capsys, "--partial")
        native = [flags for _why, flags in _answers(capsys)]

        registry._RUN_ANCHOR = None
        (tmp_path / "executor-leg").mkdir()
        other = make_repo(tmp_path / "executor-leg", monkeypatch, PLAN)
        _policy(other, ALLOW)
        monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(
            [f"T00{n}" for n in range(1, 8)]))
        monkeypatch.setenv("MOCK_LLM_WRITE_CHECK_IF_PROMPT_HAS", CHECK_GATE)
        _check(other, passed=False)
        monkeypatch.setenv("MOCK_LLM_WRITE_FAIL_IF_PROMPT_HAS", GREETING)
        import implement_orchestrator

        assert implement_orchestrator.main([
            "--plan-file", str(other.plan), "--yes", "--exec-subagents"]) == 3
        capsys.readouterr()
        report = json.loads((other.phase_dir / "write_batch_report.json").read_text())
        assert report["stop"]["task_id"] == "T003"
        assert report["stop"]["bytes_moved"] is True
        executor = [entry["flags"] for entry in report["stop"]["answers"]]
        assert executor == native == CHECK_ONLY

    def test_partial_is_refused_beside_completed(self, repo, capsys):
        _emit(repo, capsys)
        code, _out, err = _util(capsys, "batch-done", "--phase-dir", str(repo.phase_dir),
                                "--task-id", "T001", "--status", "completed", "--partial")
        assert code == 3
        assert "--partial" in err and "--status failed" in err
        assert get_or_create_state(repo.plan).state["task_status"].get("T001") != "completed"


class TestAnUnlandedReplyIsAnsweredAsTheExecutorAnswersIt:
    """A reply that said `applied` and changed no file, twice: the reason
    the instructions prescribe. The executor offers `skip` first, then
    `done` for a change made by hand — there is no failure to accept."""

    def test_skip_comes_first_and_done_is_for_a_change_made_by_hand(self, repo, capsys):
        _native_stop(repo, capsys, "--reason", "said applied, no file changed")
        pairs = _answers(capsys)
        own = [(why, flags) for why, flags in pairs if any("T003" in f for f in flags)]
        assert [flags for _why, flags in own] == [
            ["--resume", "--human-task", "T002=done", "--human-task", "T003=skip"],
            ["--resume", "--human-task", "T003=skip"],
            ["--resume", "--human-task", "T003=done"],
        ], own
        assert own[2][0] == (
            "once you have made T003's change yourself: T003 counts as done "
            "and the tasks after it run")
        assert not any("accept T003's failure" in why for why, _flags in pairs)


def test_the_core_tells_the_host_to_record_it():
    from pathlib import Path

    core = " ".join((Path(__file__).parent.parent / "instructions" / "implement.md")
                    .read_text(encoding="utf-8").split())
    assert ("and `--partial` when its subagent changed files before it failed "
            "(its reply's `files_touched`, or files you see changed on disk)") in core
    assert "Use `failed` for a check task also when its subagent changed a file, with `--partial`." in core
