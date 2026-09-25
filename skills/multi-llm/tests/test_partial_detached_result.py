#!/usr/bin/env python3
"""A partial detached run reports ``partial`` to the host (plan §5, task 5.3).

The detached child's own ``MULTI_LLM_RESULT`` said ``partial``, but the host
never reads that line: it polls ``status --wait``, whose result took its state
from the close-out record, which carried only the exit code (0 -> ``done``).
The close-out now records ``run.result_state`` (bounded to the words the exit
code cannot express), and a selector-pinned poll restates it.

These drive the real dispatcher close-out (``multi_llm.main`` ->
``_close_detached_run`` -> ``detach.finish_active_run``) under a real
``RunHandle``, with a stub target standing in for the phase.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import multi_llm  # noqa: E402
from utils import cli_common, detach, run_status, status_file, unattended  # noqa: E402

FAILED_SPEC = "codex:gpt-6-astra:xhigh"
FAILED_REASON = "no result (cancelled at the fan-out deadline)"
GOOD_SPEC = "claude:opus"


def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "a plan.md"
    plan.write_text("# Plan\n\n## Tasks\n- [ ] 1. Do the thing\n", encoding="utf-8")
    return plan


def _phase_dir(plan: Path, phase: str = "review-plan") -> Path:
    directory = plan.parent / plan.stem.replace(" ", "_") / phase
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _run_dispatcher(monkeypatch, capsys, plan, phase, run_id, target):
    """``multi_llm.main`` under a live ``RunHandle``, closed out by the dispatcher."""
    handle = detach.RunHandle(
        str(phase), run_id=run_id, mode="review-plan",
        plan_file=str(plan), detached=False,
    )
    handle.start()
    code = 1
    try:
        monkeypatch.setattr(multi_llm, "_load_target", lambda m, a: target)
        code = multi_llm.main(["review-plan", str(plan)])
        capsys.readouterr()
    finally:
        # A backstop only: the dispatcher's own close-out finished the run, and
        # `finish` is idempotent (the first call wins).
        handle.finish(code)
    return code


def _poll(plan, *extra):
    """A pinned ``status`` poll; returns (exit, result-line payload)."""
    context = cli_common.activate_result(cli_common.ResultContext(mode="status"))
    try:
        code = run_status.main(["--plan-file", str(plan), "--view", "run", *extra])
    finally:
        cli_common.deactivate_result()
    return code, context.payload(code)


def _partial_target(phase):
    def target(argv):
        progress = {
            "step": "complete", "models_total": 2, "models_done": 2,
            "models_succeeded": 1,
            "models": {
                GOOD_SPEC: {"state": "done"},
                FAILED_SPEC: {"state": "failed", "error": FAILED_REASON},
            },
        }
        status_file.update_status(str(phase), {
            "phase": "review-plan", "state": "complete",
            "models_completed": [GOOD_SPEC],
            "models_failed": {FAILED_SPEC: FAILED_REASON},
            "progress": progress,
        })
        return unattended.report_partial_results(
            completed=[GOOD_SPEC], failed=[FAILED_SPEC], phase_dir=str(phase),
        )
    return target


class TestTheCarriedWordIsBounded:
    def test_only_partial_crosses_the_process_boundary(self):
        assert cli_common.STATE_PARTIAL == "partial"
        assert detach.carried_result_state(cli_common.STATE_PARTIAL) == "partial"
        for word in (None, "done", "error", "needs-human", "nothing-to-do",
                     "running", "died", "bogus"):
            assert detach.carried_result_state(word) is None, word


class TestAPartialDetachedRunReadsPartial:
    def test_the_poll_says_partial_and_names_the_failed_slot(
        self, tmp_path, monkeypatch, capsys
    ):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        code = _run_dispatcher(
            monkeypatch, capsys, plan, phase, "run-partial-0001",
            _partial_target(phase),
        )
        assert code == 0
        run = status_file.read_run(str(phase))
        # `run.state` keeps its process meaning; the word rides beside it.
        assert run["state"] == "done"
        assert run["exit_code"] == 0
        assert run["result_state"] == cli_common.STATE_PARTIAL

        exit_code, payload = _poll(plan, "--run-id", "run-partial-0001")
        capsys.readouterr()
        assert exit_code == 0
        assert payload["exit"] == 0
        assert payload["state"] == cli_common.STATE_PARTIAL
        models = payload["progress"]["models"]
        assert models[FAILED_SPEC]["state"] == "failed"
        assert models[FAILED_SPEC]["error"] == FAILED_REASON

    def test_json_keeps_the_process_word_on_the_run(
        self, tmp_path, monkeypatch, capsys
    ):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _run_dispatcher(
            monkeypatch, capsys, plan, phase, "run-partial-0002",
            _partial_target(phase),
        )
        exit_code, payload = _poll(plan, "--run-id", "run-partial-0002", "--json")
        body = json.loads(capsys.readouterr().out)
        assert exit_code == 0
        assert body["runs"][0]["run"]["state"] == "done"
        assert payload["state"] == cli_common.STATE_PARTIAL

    def test_the_text_view_annotates_run_state(
        self, tmp_path, monkeypatch, capsys
    ):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _run_dispatcher(
            monkeypatch, capsys, plan, phase, "run-partial-0003",
            _partial_target(phase),
        )
        _poll(plan, "--run-id", "run-partial-0003")
        out = capsys.readouterr().out
        assert f"run.state:      done (partial — failed: {FAILED_SPEC})" in out

    def test_an_unpinned_poll_keeps_done(self, tmp_path, monkeypatch, capsys):
        """No selector, no single watched run: the word is not claimed."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _run_dispatcher(
            monkeypatch, capsys, plan, phase, "run-partial-0004",
            _partial_target(phase),
        )
        exit_code, payload = _poll(plan)
        capsys.readouterr()
        assert exit_code == 0
        assert payload["state"] == "done"


class TestACleanRunStaysDone:
    def test_a_clean_run_records_null_and_polls_done(
        self, tmp_path, monkeypatch, capsys
    ):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        code = _run_dispatcher(
            monkeypatch, capsys, plan, phase, "run-clean-00001", lambda argv: 0,
        )
        assert code == 0
        run = status_file.read_run(str(phase))
        assert run["state"] == "done"
        assert run["result_state"] is None
        exit_code, payload = _poll(plan, "--run-id", "run-clean-00001")
        out = capsys.readouterr().out
        assert (exit_code, payload["state"]) == (0, "done")
        assert "run.state:      done\n" in out

    @pytest.mark.parametrize("stored", ["absent", None])
    def test_a_record_written_before_this_change_polls_done(
        self, tmp_path, capsys, stored
    ):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        run = {
            "run_id": "run-legacy-0001", "state": "done", "exit_code": 0,
            "error": None, "detached": True, "pid": 1,
            "heartbeat_at": status_file.now_iso(),
            "finished_at": status_file.now_iso(),
        }
        if stored != "absent":
            run["result_state"] = stored
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete", "run": run,
        })
        exit_code, payload = _poll(plan, "--run-id", "run-legacy-0001")
        capsys.readouterr()
        assert (exit_code, payload["state"]) == (0, "done")

    def test_a_hand_edited_word_is_not_restated(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "complete",
            "run": {
                "run_id": "run-edited-0001", "state": "done", "exit_code": 0,
                "error": None, "detached": True, "pid": 1,
                "result_state": "needs-human",
                "heartbeat_at": status_file.now_iso(),
                "finished_at": status_file.now_iso(),
            },
        })
        exit_code, payload = _poll(plan, "--run-id", "run-edited-0001")
        capsys.readouterr()
        assert (exit_code, payload["state"]) == (0, "done")


class TestACrashRecordsNoResultState:
    def test_context_state_error_is_not_carried(
        self, tmp_path, monkeypatch, capsys
    ):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)

        def boom(argv):
            ctx = cli_common.active_result()
            ctx.state = "error"
            raise RuntimeError("kaboom in the phase")

        code = _run_dispatcher(
            monkeypatch, capsys, plan, phase, "run-crash-00001", boom,
        )
        assert code == 1
        run = status_file.read_run(str(phase))
        assert run["state"] == "failed"
        assert run["exit_code"] == 1
        assert run["result_state"] is None
        exit_code, payload = _poll(plan, "--run-id", "run-crash-00001")
        capsys.readouterr()
        # The exit code's own word, once — never `error` restated from the run.
        assert exit_code == 3
        assert payload["state"] == cli_common._STATE_BY_EXIT[exit_code]

    def test_partial_then_crash_is_not_reported_partial(
        self, tmp_path, monkeypatch, capsys
    ):
        """A run that went partial and then crashed exits 1: the poll's exit
        code already says the run failed, so `partial` is not restated."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        partial = _partial_target(phase)

        def target(argv):
            partial(argv)
            raise RuntimeError("crashed after the partial aggregate")

        _run_dispatcher(
            monkeypatch, capsys, plan, phase, "run-crash-00002", target,
        )
        exit_code, payload = _poll(plan, "--run-id", "run-crash-00002")
        capsys.readouterr()
        assert exit_code == 3
        assert payload["state"] != cli_common.STATE_PARTIAL
