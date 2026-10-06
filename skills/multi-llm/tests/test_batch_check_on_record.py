"""A code task whose own check fails is on record, not only on screen.

Fix round 15 (review of round 14, exec #4). On the native leg ``util
batch-done --check findings`` on a task that edits files printed
``[BATCH_CHECK_FAILED]`` and persisted nothing, so ``status``, a later resume
and review-code never saw that the task's own check fails (a check *task*
has ``task_check_findings``). It is recorded now in ``state.json``'s
``task_batch_check`` — its own key, since it holds no task back — by both
legs, and read where the check findings are read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_check_findings_hold_dependents import repo  # noqa: F401 - fixture
from tests import test_write_executor
from tests.test_write_executor import FakeState

import code_review_orchestrator  # noqa: E402
import implement_orchestrator  # noqa: E402
import multi_llm  # noqa: E402
from utils import run_status  # noqa: E402
from utils.schema_validator import validate_state_file  # noqa: E402
from utils.state_manager import (  # noqa: E402
    TASK_BATCH_CHECK_KEY,
    get_or_create_state,
    set_task_batch_check_in_state,
)

env = test_write_executor.env

FAILING = "node tests/race-logic.test.mjs: AssertionError: a skipped gate must be rejected"


def _emit(repo, capsys):
    assert implement_orchestrator.main(["--plan-file", str(repo.plan), "--yes"]) == 0
    capsys.readouterr()


def _util(capsys, *argv):
    code = multi_llm.main(["util", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _record(repo, capsys, task_id, *extra, status="completed"):
    code, out, err = _util(
        capsys, "batch-done", "--phase-dir", str(repo.phase_dir), "--task-id",
        task_id, "--status", status, *extra,
    )
    assert code == 0, (out, err)
    return out


class TestTheNativeLegRecordsIt:

    def test_batch_done_records_and_clears_it_beside_never_in_the_check_findings(
        self, repo, capsys,
    ):
        _emit(repo, capsys)
        out = _record(repo, capsys, "T001", "--own-check", "findings", "--finding", FAILING)
        assert "[BATCH_CHECK_FAILED] T001" in out
        state = get_or_create_state(repo.plan).state
        assert state[TASK_BATCH_CHECK_KEY] == {"T001": [FAILING]}
        assert "task_check_findings" not in state

        _record(repo, capsys, "T001", "--own-check", "passed")
        assert TASK_BATCH_CHECK_KEY not in get_or_create_state(repo.plan).state
        _record(repo, capsys, "T001", "--own-check", "findings", "--finding", FAILING)
        _record(repo, capsys, "T001")
        assert TASK_BATCH_CHECK_KEY not in get_or_create_state(repo.plan).state

    def test_status_and_review_code_read_it(self, repo, capsys):
        _emit(repo, capsys)
        _record(repo, capsys, "T001", "--own-check", "findings", "--finding", FAILING)

        run_status.main(["--plan-file", str(repo.plan), "--view", "run"])
        out = capsys.readouterr().out
        assert f"own check fails T001: {FAILING}" in out

        run_status.main(["--plan-file", str(repo.plan)])
        out = capsys.readouterr().out
        assert (
            "a task's own report says a check it ran fails (see "
            "implement/summary.md): T001 (\""
        ) in out

        record = code_review_orchestrator.build_implementation_record(repo.plan)
        assert (
            "- Task T001's own check failed after its edits (the edits were kept; "
            "report each failure that still holds in the code as an issue): "
            + FAILING
        ) in record

    def test_the_state_schema_admits_it(self, repo, capsys):
        _emit(repo, capsys)
        _record(repo, capsys, "T001", "--own-check", "findings", "--finding", FAILING)
        state = get_or_create_state(repo.plan).state
        valid, errors = validate_state_file(state)
        assert valid, errors
        schema = json.loads((Path(implement_orchestrator.__file__).parent / "schemas"
                             / "state_file.schema.json").read_text(encoding="utf-8"))
        assert schema["properties"][TASK_BATCH_CHECK_KEY]["additionalProperties"] == {
            "type": "array", "items": {"type": "string"}}

    def test_the_helper_drops_the_key_when_it_empties(self):
        state: dict = {}
        set_task_batch_check_in_state(state, "T001", [FAILING])
        set_task_batch_check_in_state(state, "T002", ["second"])
        set_task_batch_check_in_state(state, "T001", [])
        assert state == {TASK_BATCH_CHECK_KEY: {"T002": ["second"]}}
        set_task_batch_check_in_state(state, "T002", None)
        assert state == {}


class TestTheExecutorLegRecordsItToo:

    def _task(self, env, task_id, path, deps=()):
        return env.apply_task(
            int(task_id[1:]) - 1, kind="implement", task_id=task_id,
            expected_ids=[task_id], output_name=f"implement_batch_{task_id}.json",
            declared_files=[str(env.root / path)], depends_on=list(deps),
            prior_changes_id=task_id,
        )

    def test_a_failing_own_check_is_written_to_state(self, env):
        env.monkeypatch.setenv("MOCK_LLM_EDITS", json.dumps(
            [{"path": "src/a.py", "mode": "patch", "content": "a = 3\n"}]))
        env.monkeypatch.setenv("MOCK_LLM_WRITE_IDS", json.dumps(["T001", "T002"]))
        env.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK", json.dumps(
            {"passed": False, "findings": [FAILING]}))
        env.monkeypatch.setenv("MOCK_LLM_WRITE_BATCH_CHECK_IF_PROMPT_HAS", "Apply batch 0")
        state = FakeState(env.tmp_path / "state.json")
        result = env.scheduler(
            [self._task(env, "T001", "src/a.py"),
             self._task(env, "T002", "src/a.py", deps=["T001"])],
            kind="implement", state=state,
            summary_file=str(env.out_dir / "implement" / "summary.md"),
        ).run()
        assert result.exit_code == 0, result.messages
        assert state.state[TASK_BATCH_CHECK_KEY] == {"T001": [FAILING]}
        assert "task_check_findings" not in state.state
