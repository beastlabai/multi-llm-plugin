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
from utils.json_extractor import sanitize_model_name  # noqa: E402

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


# ---------------------------------------------------------------------------
# Plan review-run-scope-salvage-and-failure-visibility §5 (T5.4a/b/c): a
# reaggregate tells the host which reviewers it still has no result for
# ---------------------------------------------------------------------------
#
# These drive the real `--reaggregate` through the dispatcher — no stub
# target — under a `RunHandle`, whose claim drops `progress` wholesale exactly
# as a detached worker's does. That is the case the Codex templates could not
# serve: "name the failed models from `progress.models`" read an empty map.

SUGGESTION = {
    "title": "Add input validation",
    "desc": "The plan does not validate the new email field.",
    "importance": "HIGH",
    "reference": "Overview",
    "type": "addition",
}


def _left_by_a_fanout(phase: Path, *, good: bool = True) -> None:
    """The dir a lossy fan-out leaves: one result file, one recorded failure."""
    requested = [GOOD_SPEC, FAILED_SPEC] if good else [FAILED_SPEC]
    if good:
        (phase / f"{sanitize_model_name(GOOD_SPEC)}.json").write_text(
            json.dumps([SUGGESTION]), encoding="utf-8"
        )
    status_file.write_status(str(phase), {
        "phase": "review-plan",
        "state": "complete",
        "started_at": "2026-09-25T15:58:00",
        "models_requested": requested,
        "models_completed": [GOOD_SPEC] if good else [],
        "models_failed": {FAILED_SPEC: FAILED_REASON},
        "progress": {
            "step": "complete", "models_total": len(requested),
            "models_done": len(requested), "models_succeeded": int(good),
            "models": {
                **({GOOD_SPEC: {"state": "done"}} if good else {}),
                FAILED_SPEC: {"state": "failed", "error": FAILED_REASON},
            },
        },
    })


def _reaggregate_under_a_run(capsys, plan, phase, run_id, *argv):
    """The real reaggregate, dispatched in-process under a claimed run."""
    handle = detach.RunHandle(
        str(phase), run_id=run_id, mode=argv[0], plan_file=str(plan),
        detached=False,
    )
    handle.start()
    # The claim is what a detached worker does first: `progress` is gone.
    assert "progress" not in status_file.read_status(str(phase)) or not (
        status_file.read_status(str(phase)).get("progress") or {}
    ).get("models")
    code = 1
    try:
        code = multi_llm.main([argv[0], str(plan), *argv[1:]])
    finally:
        handle.finish(code)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _phase_recorded_completed(plan: Path, phase_name: str) -> bool:
    from utils.state_manager import get_or_create_state

    return get_or_create_state(plan).is_phase_completed(phase_name)


class TestAReaggregateWithAFailureLeftReportsPartial:
    def test_it_names_the_model_completes_the_phase_and_owes_nothing(
        self, tmp_path, capsys
    ):
        """T5.4a: holds with or without T5.4b."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _left_by_a_fanout(phase)

        code, out, _err = _reaggregate_under_a_run(
            capsys, plan, phase, "run-reagg-00001",
            "review-plan", "--reaggregate", "--skip-validation",
        )

        assert code == 0
        assert f"[MODEL_FAILED] {FAILED_SPEC}: {FAILED_REASON}" in out
        assert _phase_recorded_completed(plan, "review-plan")
        assert status_file.read_status(str(phase))["state"] == "complete"
        result = cli_common.parse_result_line(out)
        assert result["next"] is None
        report = (phase / "report.md").read_text(encoding="utf-8")
        assert f"**Models failed:** 1 of 2 requested — {FAILED_SPEC}: {FAILED_REASON}" in report

    def test_the_run_reads_partial_and_progress_names_the_slot(
        self, tmp_path, capsys
    ):
        """T5.4b: `[PARTIAL_RESULTS]`, `state: partial` on the line and on the
        poll, and the failed row written back although the claim dropped
        `progress` — with the phase still completed and `next: null`."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _left_by_a_fanout(phase)

        code, out, _err = _reaggregate_under_a_run(
            capsys, plan, phase, "run-reagg-00002",
            "review-plan", "--reaggregate", "--skip-validation",
        )

        assert code == 0
        assert f"[PARTIAL_RESULTS: {FAILED_SPEC}]" in out
        result = cli_common.parse_result_line(out)
        assert result["state"] == cli_common.STATE_PARTIAL
        assert result["exit"] == 0
        assert result["next"] is None
        assert status_file.read_run(str(phase))["result_state"] == cli_common.STATE_PARTIAL

        exit_code, payload = _poll(plan, "--run-id", "run-reagg-00002")
        capsys.readouterr()
        assert exit_code == 0
        assert payload["state"] == cli_common.STATE_PARTIAL
        row = payload["progress"]["models"][FAILED_SPEC]
        assert row["state"] == "failed"
        assert row["error"] == FAILED_REASON
        assert _phase_recorded_completed(plan, "review-plan")

    def test_a_clean_reaggregate_stays_done(self, tmp_path, capsys):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _left_by_a_fanout(phase)
        (phase / f"{sanitize_model_name(FAILED_SPEC)}.json").write_text(
            json.dumps([SUGGESTION]), encoding="utf-8"
        )

        code, out, _err = _reaggregate_under_a_run(
            capsys, plan, phase, "run-reagg-00003",
            "review-plan", "--reaggregate", "--skip-validation",
        )

        assert code == 0
        assert "[MODEL_FAILED]" not in out
        assert "[PARTIAL_RESULTS:" not in out
        assert cli_common.parse_result_line(out)["state"] == "done"


class TestAReaggregateWhoseAnswersHoldNoFindingsStillCompletes:
    """A reviewer that answered ``[]`` beside one that failed is a partial
    review with zero groups, not "nothing to aggregate": the reaggregate
    rebuilds the report naming the failure, reports ``partial`` and records
    the phase completed, exactly as it does when the answer held findings."""

    @pytest.mark.parametrize("mode", ["review-plan", "review-tasks"])
    @pytest.mark.parametrize("extra", [["--skip-validation"], []])
    def test_the_failure_is_reported_and_the_phase_completes(
        self, tmp_path, capsys, mode, extra
    ):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, mode)
        _left_by_a_fanout(phase)
        (phase / f"{sanitize_model_name(GOOD_SPEC)}.json").write_text(
            "[]", encoding="utf-8"
        )

        code, out, _err = _reaggregate_under_a_run(
            capsys, plan, phase, "run-reagg-00006", mode, "--reaggregate", *extra,
        )

        assert code == 0
        assert f"[MODEL_FAILED] {FAILED_SPEC}: {FAILED_REASON}" in out
        assert f"[PARTIAL_RESULTS: {FAILED_SPEC}]" in out
        assert "[VALIDATION" not in out
        result = cli_common.parse_result_line(out)
        assert (result["exit"], result["state"]) == (0, cli_common.STATE_PARTIAL)
        assert result["next"] is None
        assert _phase_recorded_completed(plan, mode)
        assert status_file.read_status(str(phase))["state"] == "complete"
        report = (phase / "report.md").read_text(encoding="utf-8")
        assert f"**Models failed:** 1 of 2 requested — {FAILED_SPEC}: {FAILED_REASON}" in report
        assert "**Groups:** 0" in report
        assert "_No suggestions found._" in report

    def test_an_earlier_rounds_grouping_is_not_reported_or_validated(
        self, tmp_path, capsys
    ):
        """The grouping and validation round left by an earlier run describe
        findings no longer on disk: the report shows zero groups, and the
        leftover task file is neither read against the new grouping nor
        re-materialized. It retires with the rest of its round, and
        ``validation.json`` is written as ``[]``, so apply-suggestions loads
        zero groups (plan improve-plan-and-improve-code T3; this test used to
        pin the task file left in place)."""
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        _left_by_a_fanout(phase)
        (phase / f"{sanitize_model_name(GOOD_SPEC)}.json").write_text(
            "[]", encoding="utf-8"
        )
        stale_group = {
            "theme": "Add input validation",
            "suggestions": [{**SUGGESTION, "source_model": GOOD_SPEC}],
        }
        (phase / "grouped.json").write_text(
            json.dumps({"format_version": 2, "groups": [stale_group]}),
            encoding="utf-8",
        )
        stale_tasks = json.dumps({"tasks_schema_version": 2, "batches": []})
        (phase / "validation_tasks.json").write_text(stale_tasks, encoding="utf-8")

        code, out, _err = _reaggregate_under_a_run(
            capsys, plan, phase, "run-reagg-00007", "review-plan", "--reaggregate",
        )

        assert code == 0
        assert "[VALIDATION_PENDING]" not in out
        assert "[VALIDATION_BATCHES_PENDING]" not in out
        grouped = json.loads((phase / "grouped.json").read_text(encoding="utf-8"))
        assert grouped["groups"] == []
        assert not (phase / "validation_tasks.json").exists()
        retired = sorted(phase.glob("validation_tasks.*.stale.json"))
        assert [p.read_text(encoding="utf-8") for p in retired] == [stale_tasks]
        assert (phase / "validation.json").read_text(encoding="utf-8") == "[]\n"
        report = (phase / "report.md").read_text(encoding="utf-8")
        assert "**Groups:** 0" in report
        assert "Add input validation" not in report
        assert _phase_recorded_completed(plan, "review-plan")


class TestAReaggregateWhereEveryReviewerFailedExitsThree:
    @pytest.mark.parametrize("mode, phase_name", [
        ("review-plan", "review-plan"),
        ("review-code", "code-review"),
    ])
    def test_it_exits_three_names_the_model_and_records_no_phase(
        self, tmp_path, capsys, mode, phase_name
    ):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan, phase_name)
        _left_by_a_fanout(phase, good=False)

        code, out, err = _reaggregate_under_a_run(
            capsys, plan, phase, "run-reagg-00004", mode, "--reaggregate",
        )

        assert code == 3
        assert f"[MODEL_FAILED] {FAILED_SPEC}: {FAILED_REASON}" in out
        assert "no model produced a result" in err
        assert FAILED_SPEC in err
        assert "--force" in err
        result = cli_common.parse_result_line(out)
        assert (result["exit"], result["state"]) == (3, "needs-human")
        assert not _phase_recorded_completed(plan, phase_name)

    def test_no_results_and_no_recorded_failure_still_exits_zero(
        self, tmp_path, capsys
    ):
        plan = _plan(tmp_path)
        phase = _phase_dir(plan)
        status_file.write_status(str(phase), {
            "phase": "review-plan", "state": "models_running",
            "models_requested": [GOOD_SPEC],
        })

        code, out, _err = _reaggregate_under_a_run(
            capsys, plan, phase, "run-reagg-00005",
            "review-plan", "--reaggregate", "--skip-validation",
        )

        assert code == 0
        assert "No model result files found" in out
        assert not _phase_recorded_completed(plan, "review-plan")

    def test_a_full_run_that_hits_the_three_stops_and_relays_it(
        self, tmp_path, monkeypatch, capsys
    ):
        """`full` never re-dispatches a phase that exited 3: it stops there,
        reports the phase and the `--resume` re-invocation, and exits 3."""
        import full_workflow_orchestrator as fwo
        from utils.state_manager import StateManager

        plan = _plan(tmp_path)
        _left_by_a_fanout(_phase_dir(plan, "code-review"), good=False)
        state = StateManager(plan)
        for done in ("review-plan", "apply-suggestions", "generate-tasks",
                     "review-tasks", "apply-task-suggestions", "implement"):
            state.mark_phase_completed(done)
        state.save()

        real_run_mode = fwo.FullWorkflow.run_mode
        calls = []

        def run_mode(self, mode_word, extra):
            calls.append(mode_word)
            assert mode_word == "review-code", f"no other phase may run: {mode_word}"
            # The phase's own code path, ending in the reaggregate's exit 3.
            return real_run_mode(self, mode_word, [*extra, "--reaggregate"])

        monkeypatch.setattr(fwo.FullWorkflow, "run_mode", run_mode)
        code = multi_llm.main(["full", str(plan), "--yes"])
        captured = capsys.readouterr()

        assert code == 3
        assert calls == ["review-code"], "stopped, not retried"
        assert "Phase 5: Review Code did not complete (exited 3)" in captured.err
        assert "--resume" in captured.err
        result = cli_common.parse_result_line(captured.out)
        assert (result["exit"], result["state"]) == (3, "needs-human")
        assert not StateManager(plan).is_phase_completed("code-review")
