"""A reviewer's result file under the wrong name (live run 10, Claude Code racing 07).

The pi reviewer ``pi:openrouter/deepseek/deepseek-v4.1-flash`` was asked to write
``pi_openrouter_deepseek_deepseek-v4_1-flash.json`` and wrote its three findings
to ``pi_openrouter_deepseek_deepseek-v4-flash.json`` (``v4``, not ``v4_1``). Two
things then went wrong, and each half of the fix is pinned here on that shape:

(a) the slot read only the requested name, found nothing ("File not found"), and
    recorded the model failed although its answer was on disk. A slot whose
    requested file is missing now reads the one fresh JSON file its own provider's
    prefix wrote during the run, says so, and renames it to the requested name so
    nothing reads it twice. Anything less certain (two candidates, another model
    of the same provider in the run, an old file, a non-list) leaves the model
    failed as before.

(b) ``--reaggregate`` globbed ``*.json`` and loaded the misnamed file as a third
    reviewer named by its stem, with "scope unknown": its findings were mixed into
    report.md and the first validation round was thrown away. The aggregate now
    reads only the files of the models the phase's ``.status.json`` records; any
    other result file is named in a note and set aside (renamed, never deleted).
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

import code_review_orchestrator as cro  # noqa: E402
from utils import result_files  # noqa: E402
from utils import review_orchestrator_base as rob  # noqa: E402
from utils.llm_client import ERROR_PARSE_ERROR, ERROR_TIMEOUT, invoke_with_file_output  # noqa: E402
from utils.status_file import read_status, write_status  # noqa: E402

from tests.test_reaggregate_identity import (  # noqa: E402
    _cro_phase,
    _cro_report_md,
    _phase,
    _reaggregate,
    _reported_results,
    _run_cro_reaggregate,
    _suggestions,
)

PI = "pi:openrouter/deepseek/deepseek-v4.1-flash"
CODEX = "codex:gpt-6-luna:high"
PI_STEM = "pi_openrouter_deepseek_deepseek-v4_1-flash"
CODEX_STEM = "codex_gpt-6-luna_high"
#: The name the model really wrote (live run 10).
MISNAMED = "pi_openrouter_deepseek_deepseek-v4-flash.json"


def _issues(titles):
    return [
        {
            "title": title,
            "desc": f"Description for {title}",
            "importance": "medium",
            "file": "index.html",
            "line_range": [1, 2],
            "type": "bug",
            "anchor_text": "x",
        }
        for title in titles
    ]


def _parse_failure():
    """What ``invoke_with_provider`` returned for the live pi slot: exit 0, no JSON."""
    return {
        "success": False,
        "error": "No valid JSON in stdout",
        "error_code": ERROR_PARSE_ERROR,
        "data": None,
        "details": {"provider": "pi", "model": "openrouter/deepseek/deepseek-v4.1-flash"},
    }


def _invoke(tmp_path, writes, run_model_specs=(PI, CODEX), *, model_spec=PI):
    """One ``invoke_with_file_output`` call whose provider writes *writes*.

    ``writes`` maps a file name in the phase dir to its JSON payload; the files
    are written *during* the provider call, the way the CLI writes them.
    """
    phase_dir = tmp_path / "code-review"

    def provider(*_args, **_kwargs):
        for name, payload in writes.items():
            (phase_dir / name).write_text(json.dumps(payload), encoding="utf-8")
        return _parse_failure()

    with patch("utils.llm_client.invoke_with_provider", side_effect=provider):
        return invoke_with_file_output(
            prompt_template="Write to {output_json_path}",
            model_spec=model_spec,
            prompt_context={},
            output_dir=tmp_path,
            phase="code_review",
            timeout=600,
            run_model_specs=None if run_model_specs is None else list(run_model_specs),
        ), phase_dir


# ---------------------------------------------------------------------------
# (a) the slot reads the answer from where the model put it
# ---------------------------------------------------------------------------

class TestAMisnamedResultIsReadFromWhereTheModelWroteIt:
    def test_the_live_shape_is_read_and_renamed_to_the_requested_name(self, tmp_path):
        findings = _issues(["Curvature", "Lane offset", "resetCars"])
        result, phase_dir = _invoke(tmp_path, {MISNAMED: findings})

        assert result["success"] is True
        assert result["data"] == findings
        assert result["misnamed_from"] == MISNAMED
        # Renamed, so the file is the model's own from here on and nothing can
        # read it a second time under the other name.
        requested = phase_dir / f"{PI_STEM}.json"
        assert result["output_file"] == str(requested)
        assert json.loads(requested.read_text(encoding="utf-8")) == findings
        assert not (phase_dir / MISNAMED).exists()

    def test_the_slot_log_says_where_the_result_came_from(self, tmp_path):
        phase_dir = tmp_path / "code-review"
        log = tmp_path / "log_pi.txt"

        def provider(*_args, **_kwargs):
            (phase_dir / MISNAMED).write_text(json.dumps(_issues(["A"])), encoding="utf-8")
            failure = _parse_failure()
            failure["details"]["_deferred_log"] = {
                "stdout": "prose", "stderr": "", "returncode": 0, "duration_seconds": 1.0,
            }
            return failure

        with patch("utils.llm_client.invoke_with_provider", side_effect=provider):
            invoke_with_file_output(
                prompt_template="Write to {output_json_path}", model_spec=PI,
                prompt_context={}, output_dir=tmp_path, phase="code_review",
                timeout=600, log_file=str(log), run_model_specs=[PI, CODEX],
            )
        text = log.read_text(encoding="utf-8")
        assert f"Result source: file (written as {MISNAMED})" in text
        assert "Success: True" in text
        assert "Stdout: not used (file output)" in text

    def test_the_note_names_both_files(self):
        assert result_files.misnamed_result_note(PI, MISNAMED, f"{PI_STEM}.json") == (
            f"{PI} wrote its result to {MISNAMED} instead of {PI_STEM}.json; "
            f"read it from there and renamed it to {PI_STEM}.json"
        )

    def test_without_the_runs_model_list_nothing_is_adopted(self, tmp_path):
        """The opt-in: a caller that does not say which models share the dir
        keeps today's behaviour exactly."""
        result, phase_dir = _invoke(
            tmp_path, {MISNAMED: _issues(["A"])}, run_model_specs=None
        )
        assert result["success"] is False
        assert (phase_dir / MISNAMED).exists()

    def test_another_model_of_the_same_provider_makes_it_ambiguous(self, tmp_path):
        other_pi = "pi:openrouter/qwen/qwen3-max"
        result, phase_dir = _invoke(
            tmp_path, {MISNAMED: _issues(["A"])},
            run_model_specs=(PI, other_pi, CODEX),
        )
        assert result["success"] is False
        assert "misnamed_from" not in result
        assert (phase_dir / MISNAMED).exists()

    def test_two_fresh_candidates_are_ambiguous(self, tmp_path):
        result, phase_dir = _invoke(tmp_path, {
            MISNAMED: _issues(["A"]),
            "pi_deepseek-v4-flash.json": _issues(["B"]),
        })
        assert result["success"] is False
        assert (phase_dir / MISNAMED).exists()

    def test_a_file_that_was_there_before_the_run_is_not_this_runs(self, tmp_path):
        phase_dir = tmp_path / "code-review"
        phase_dir.mkdir(parents=True)
        (phase_dir / MISNAMED).write_text(json.dumps(_issues(["Old"])), encoding="utf-8")

        result, _ = _invoke(tmp_path, {})

        assert result["success"] is False
        assert (phase_dir / MISNAMED).exists()

    def test_another_providers_prefix_is_not_this_models(self, tmp_path):
        result, phase_dir = _invoke(tmp_path, {"omp_deepseek-v4-flash.json": _issues(["A"])})
        assert result["success"] is False
        assert (phase_dir / "omp_deepseek-v4-flash.json").exists()

    def test_another_requested_models_file_is_never_taken(self, tmp_path):
        """codex's own file appearing during pi's run is codex's, whatever the
        timing (the two run side by side)."""
        result, phase_dir = _invoke(
            tmp_path, {f"{CODEX_STEM}.json": _issues(["A"])},
            model_spec=PI, run_model_specs=(PI, CODEX),
        )
        assert result["success"] is False
        assert (phase_dir / f"{CODEX_STEM}.json").exists()

    def test_a_fresh_file_that_is_not_a_list_is_not_adopted(self, tmp_path):
        result, phase_dir = _invoke(tmp_path, {MISNAMED: {"title": "not a list"}})
        assert result["success"] is False
        assert (phase_dir / MISNAMED).exists()

    def test_a_link_under_a_result_name_is_never_followed(self, tmp_path):
        target = tmp_path / "elsewhere.json"
        target.write_text(json.dumps(_issues(["Planted"])), encoding="utf-8")
        phase_dir = tmp_path / "code-review"

        def provider(*_args, **_kwargs):
            (phase_dir / MISNAMED).symlink_to(target)
            return _parse_failure()

        with patch("utils.llm_client.invoke_with_provider", side_effect=provider):
            result = invoke_with_file_output(
                prompt_template="Write to {output_json_path}", model_spec=PI,
                prompt_context={}, output_dir=tmp_path, phase="code_review",
                timeout=600, run_model_specs=[PI, CODEX],
            )
        assert result["success"] is False
        assert (phase_dir / MISNAMED).is_symlink()

    def test_the_requested_file_still_wins_when_both_exist(self, tmp_path):
        mine = _issues(["Mine"])
        result, phase_dir = _invoke(tmp_path, {
            f"{PI_STEM}.json": mine,
            MISNAMED: _issues(["Other"]),
        })
        assert result["success"] is True
        assert result["data"] == mine
        assert "misnamed_from" not in result
        assert (phase_dir / MISNAMED).exists()


def _no_sleep(_delay):
    async def _noop():
        return None
    return _noop()


class TestAnEarlierAttemptsMisnamedFileIsStillThisSlots:
    """Review 24 (records #1): the baseline a misnamed file must be fresh against
    was taken inside each ``invoke_with_file_output`` call, so once per ladder
    attempt. Attempt 1 wrote the misnamed array and was killed at its timeout;
    attempt 2's own baseline already held the file, the slot was recorded failed,
    and the next ``--reaggregate`` set the findings aside as "not the result file
    of any model this review ran". The baseline is the slot's now: taken once,
    before the first attempt, and handed to every attempt."""

    def test_a_retry_reads_the_file_the_first_attempt_wrote(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = rob.get_phase_dir(plan, "review-plan")
        findings = _issues(["Curvature", "Lane offset"])
        attempts = []

        def provider(*_args, **_kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                # Writes the array under the wrong name, then is killed.
                (phase_dir / MISNAMED).write_text(json.dumps(findings), encoding="utf-8")
                return {
                    "success": False,
                    "error": "pi timed out after 600s",
                    "error_code": ERROR_TIMEOUT,
                    "details": {"provider": "pi"},
                }
            # The retry writes nothing and answers in prose.
            return _parse_failure()

        with patch("utils.llm_client.invoke_with_provider", side_effect=provider), \
                patch.object(rob, "get_subagent_retries", return_value=1), \
                patch.object(rob.asyncio, "sleep", _no_sleep):
            result = asyncio.run(rob.run_single_model(
                PI, str(plan), "review {output_json_path}", {},
                "review-plan", "plan_review",
                timeout=1800, out_dir=str(phase_dir.parent), prefix="plan",
                run_model_specs=[PI, CODEX],
            ))

        assert len(attempts) == 2
        assert result["success"] is True, result.get("error")
        assert json.loads(result["output"]) == findings
        assert json.loads((phase_dir / f"{PI_STEM}.json").read_text(encoding="utf-8")) == findings
        assert not (phase_dir / MISNAMED).exists()

    def test_a_file_older_than_the_slot_is_still_not_this_slots(self, tmp_path):
        """The slot-wide baseline keeps the freshness guard: a file that was in
        the dir before the first attempt is never adopted on any attempt."""
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = rob.get_phase_dir(plan, "review-plan")
        (phase_dir / MISNAMED).write_text(json.dumps(_issues(["Old"])), encoding="utf-8")
        attempts = []

        def provider(*_args, **_kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                return {"success": False, "error": "pi timed out after 600s",
                        "error_code": ERROR_TIMEOUT, "details": {"provider": "pi"}}
            return _parse_failure()

        with patch("utils.llm_client.invoke_with_provider", side_effect=provider), \
                patch.object(rob, "get_subagent_retries", return_value=1), \
                patch.object(rob.asyncio, "sleep", _no_sleep):
            result = asyncio.run(rob.run_single_model(
                PI, str(plan), "review {output_json_path}", {},
                "review-plan", "plan_review",
                timeout=1800, out_dir=str(phase_dir.parent), prefix="plan",
                run_model_specs=[PI, CODEX],
            ))

        assert len(attempts) == 2
        assert result["success"] is False
        assert (phase_dir / MISNAMED).exists()

    def test_every_attempt_is_handed_the_one_baseline(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = rob.get_phase_dir(plan, "review-plan")
        (phase_dir / "pi_old.json").write_text("[]", encoding="utf-8")
        seen = []

        def fake_invoke(**kwargs):
            seen.append(kwargs.get("misnamed_baseline"))
            # Each attempt leaves a new file: a per-attempt baseline would grow.
            (phase_dir / f"pi_attempt{len(seen)}.json").write_text("[]", encoding="utf-8")
            return {"success": False, "error": "pi timed out after 600s",
                    "error_code": ERROR_TIMEOUT, "details": {}}

        with patch.object(rob, "invoke_with_file_output", fake_invoke), \
                patch.object(rob, "get_subagent_retries", return_value=1), \
                patch.object(rob.asyncio, "sleep", _no_sleep):
            asyncio.run(rob.run_single_model(
                PI, str(plan), "review {output_json_path}", {},
                "review-plan", "plan_review",
                timeout=1800, out_dir=str(phase_dir.parent), prefix="plan",
                run_model_specs=[PI, CODEX],
            ))

        assert len(seen) == 2
        assert seen[0] is not None and set(seen[0]) == {"pi_old.json"}
        assert seen[1] == seen[0]

    def test_the_code_review_slot_takes_its_baseline_before_the_dispatch(self, tmp_path):
        """One dispatch per slot on this lane, but the same rule: the slot takes
        the baseline and hands it to ``invoke_with_file_output``."""
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = cro.get_phase_dir(plan, "code-review")
        (phase_dir / "pi_old.json").write_text("[]", encoding="utf-8")
        spy_result = {"success": False, "error": "x", "error_code": ERROR_PARSE_ERROR,
                      "details": {}}

        with patch.object(cro, "invoke_with_file_output", return_value=spy_result) as spy:
            asyncio.run(cro.run_single_review(
                PI, plan, phase_dir.parent, run_model_specs=[PI, CODEX],
            ))

        baseline = spy.call_args.kwargs["misnamed_baseline"]
        assert set(baseline) == {"pi_old.json"}

    def test_without_the_runs_model_list_no_baseline_is_taken(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = cro.get_phase_dir(plan, "code-review")
        spy_result = {"success": False, "error": "x", "error_code": ERROR_PARSE_ERROR,
                      "details": {}}
        with patch.object(cro, "invoke_with_file_output", return_value=spy_result) as spy:
            asyncio.run(cro.run_single_review(PI, plan, phase_dir.parent))
        assert spy.call_args.kwargs.get("misnamed_baseline") is None


class TestTheReviewLanesSayWhatTheyRead:
    def _misnamed_stub(self, tmp_path, phase_dir):
        return {
            "success": True,
            "data": _issues(["A"]),
            "source": "file",
            "output_file": str(phase_dir / f"{PI_STEM}.json"),
            "misnamed_from": MISNAMED,
            "details": {"provider": "pi", "duration_seconds": 1.0},
        }

    def test_review_code_prints_and_records_the_note(self, tmp_path, capsys):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = cro.get_phase_dir(plan, "code-review")
        phase_dir.mkdir(parents=True, exist_ok=True)
        write_status(str(phase_dir), {"phase": "code-review", "models_requested": [PI, CODEX]})
        stub = self._misnamed_stub(tmp_path, phase_dir)

        with patch.object(cro, "invoke_with_file_output", return_value=stub) as spy:
            success, issues, _error = asyncio.run(cro.run_single_review(
                PI, plan, tmp_path, run_model_specs=[PI, CODEX],
            ))

        assert success is True and len(issues) == 1
        assert spy.call_args.kwargs["run_model_specs"] == [PI, CODEX]
        note = result_files.misnamed_result_note(PI, MISNAMED, f"{PI_STEM}.json")
        assert f"[RESULT_FILE_MISNAMED] {note}" in capsys.readouterr().out
        status = read_status(str(phase_dir))
        assert status["result_notes"] == [note]
        # `progress.notes` is what `status` prints and a terminal poll carries
        # into its `notice`: the only result line a tool route reads.
        assert status["progress"]["notes"] == [note]

    def test_review_plan_prints_and_records_the_note(self, tmp_path, capsys):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        phase_dir = rob.get_phase_dir(plan, "review-plan")
        phase_dir.mkdir(parents=True, exist_ok=True)
        stub = self._misnamed_stub(tmp_path, phase_dir)

        with patch("utils.review_orchestrator_base.invoke_with_file_output",
                   return_value=stub) as spy:
            result = asyncio.run(rob.run_single_model(
                PI, str(plan), "review {output_json_path}", {},
                "review-plan", "plan_review",
                timeout=1800, out_dir=str(tmp_path), prefix="plan",
                run_model_specs=[PI, CODEX],
            ))

        assert result["success"] is True
        assert spy.call_args.kwargs["run_model_specs"] == [PI, CODEX]
        note = result_files.misnamed_result_note(PI, MISNAMED, f"{PI_STEM}.json")
        assert f"[RESULT_FILE_MISNAMED] {note}" in capsys.readouterr().out
        assert read_status(str(phase_dir))["result_notes"] == [note]

    def test_run_all_reviews_hands_every_slot_the_whole_panel(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        spy = AsyncMock(return_value=(True, [], None))
        with patch.object(cro, "run_single_review", spy):
            asyncio.run(cro.run_all_reviews(
                [PI, CODEX], plan, tmp_path, timeout=600, max_parallel=2,
                skip_existing=False,
            ))
        assert {call.kwargs["run_model_specs"] == [PI, CODEX] for call in spy.call_args_list} == {True}
        assert spy.call_count == 2

    def test_run_all_models_hands_every_slot_the_whole_panel(self, tmp_path):
        plan = tmp_path / "plan.md"
        plan.write_text("# plan\n", encoding="utf-8")
        spy = AsyncMock(return_value={"success": True, "output": "[]", "source": "file"})
        with patch.object(rob, "run_single_model", spy):
            asyncio.run(rob.run_all_models(
                [PI, CODEX], str(plan), "review {output_json_path}",
                lambda _spec: {}, "review-plan", "plan_review",
                timeout=600, max_parallel=2, out_dir=str(tmp_path), prefix="plan",
                skip_existing=False,
            ))
        assert spy.call_count == 2
        assert {tuple(call.kwargs["run_model_specs"]) for call in spy.call_args_list} == {(PI, CODEX)}


# ---------------------------------------------------------------------------
# (b) the aggregate reads only the models the phase records
# ---------------------------------------------------------------------------

def _live_code_review_dir(tmp_path, *, polluted=False):
    """The phase dir live run 10 left after the host's ``--force`` retry.

    ``polluted=True`` is the checkpoint the first ``--reaggregate`` then wrote:
    the misnamed file's stem appended to ``models_completed``.
    """
    plan, phase_dir = _cro_phase(tmp_path, {
        CODEX_STEM: _issues(["Off-road shortcut"]),
        PI_STEM: _issues(["totalProgress drop", "Dead curvature rule"]),
        MISNAMED[: -len(".json")]: _issues(["Curvature", "Lane offset", "resetCars"]),
    }, models_requested=[PI, CODEX])
    if polluted:
        status = read_status(str(phase_dir))
        status["models_completed"] = [PI, CODEX, MISNAMED[: -len(".json")]]
        write_status(str(phase_dir), status)
    return plan, phase_dir


class TestTheAggregateReadsOnlyTheModelsThePhaseRecords:
    @pytest.mark.parametrize("polluted", [False, True])
    def test_the_live_shape_sets_the_stray_aside(self, tmp_path, capsys, polluted):
        plan, phase_dir = _live_code_review_dir(tmp_path, polluted=polluted)

        report = _run_cro_reaggregate(plan, phase_dir)
        out = capsys.readouterr().out

        assert sorted(_reported_results(report)) == sorted([PI, CODEX])
        issues = json.loads(
            cro.get_output_paths(plan, "code_review_issues", phase="code-review")
            .with_suffix(".json").read_text(encoding="utf-8")
        )
        assert len(issues) == 3
        # The live run's "scope unknown" third reviewer is gone.
        assert f"[{MISNAMED[: -len('.json')]}]" not in out
        # Set aside, never deleted, and no longer a `*.json` anything globs.
        assert not (phase_dir / MISNAMED).exists()
        aside = phase_dir / f"{MISNAMED}.{result_files.SET_ASIDE_SUFFIX}"
        assert json.loads(aside.read_text(encoding="utf-8"))[0]["title"] == "Curvature"
        note = result_files.set_aside_note(MISNAMED, aside.name, [PI, CODEX])
        assert f"[RESULT_SET_ASIDE] {note}" in out
        status = read_status(str(phase_dir))
        assert status["result_notes"] == [note]
        assert status["progress"]["notes"] == [note]
        assert MISNAMED[: -len(".json")] not in status["models_completed"]

    @pytest.mark.parametrize("stray_row", [True, False])
    def test_a_forgotten_stem_leaves_status_agreeing_with_the_report(
        self, tmp_path, capsys, stray_row
    ):
        """Review 24 (records #2): a foreground reaggregate of a dir a
        pre-round-24 reaggregate had touched dropped the stray stem from
        ``models_completed`` but kept its ``progress.models`` row and the
        ``models_succeeded`` that counted it, so ``status`` read "3 ok" beside a
        report of two models."""
        from utils.run_status import _progress_text

        plan, phase_dir = _live_code_review_dir(tmp_path, polluted=True)
        stem = MISNAMED[: -len(".json")]
        rows = {
            PI: {"state": "done", "source": "file"},
            CODEX: {"state": "done", "source": "file"},
        }
        if stray_row:
            rows[stem] = {"state": "done", "source": "existing_file"}
        status = read_status(str(phase_dir))
        status["progress"] = {
            "models_total": 2, "models_done": 3 if stray_row else 2,
            "models_succeeded": 3, "models": rows,
        }
        write_status(str(phase_dir), status)

        _run_cro_reaggregate(plan, phase_dir)
        capsys.readouterr()

        progress = read_status(str(phase_dir))["progress"]
        assert sorted(progress["models"]) == sorted([PI, CODEX])
        assert progress["models_succeeded"] == 2
        assert progress["models_done"] == 2
        assert "models 2/2 done, 2 ok, 0 failed" in _progress_text(progress)

    def test_a_forgotten_stem_without_a_map_recounts_the_completed_models(self, tmp_path):
        """No per-model map: ``models_succeeded`` is what the pre-24
        ``record_models_present`` wrote it as, the length of ``models_completed``."""
        _plan, phase_dir = _live_code_review_dir(tmp_path, polluted=True)
        status = read_status(str(phase_dir))
        status["progress"] = {"models_total": 2, "models_done": 2, "models_succeeded": 3}
        write_status(str(phase_dir), status)

        result_files._forget_completed(
            str(phase_dir), [MISNAMED[: -len(".json")]], [PI, CODEX]
        )

        progress = read_status(str(phase_dir))["progress"]
        assert progress["models_succeeded"] == 2
        assert progress["models_done"] == 2

    def test_a_set_aside_that_was_never_recorded_leaves_progress_alone(self, tmp_path, capsys):
        plan, phase_dir = _live_code_review_dir(tmp_path)
        status = read_status(str(phase_dir))
        status["progress"] = {
            "models_total": 2, "models_done": 2, "models_succeeded": 1,
            "models": {PI: {"state": "failed"}, CODEX: {"state": "done"}},
        }
        write_status(str(phase_dir), status)

        _run_cro_reaggregate(plan, phase_dir)
        capsys.readouterr()

        progress = read_status(str(phase_dir))["progress"]
        assert sorted(progress["models"]) == sorted([PI, CODEX])

    def test_the_report_names_the_file_it_left_out(self, tmp_path, capsys):
        plan, phase_dir = _live_code_review_dir(tmp_path)

        _run_cro_reaggregate(plan, phase_dir, render_report=True)
        capsys.readouterr()

        report = _cro_report_md(plan)
        models_line = next(ln for ln in report.splitlines() if ln.startswith("**Models:**"))
        assert MISNAMED[: -len(".json")] not in models_line
        note = result_files.set_aside_note(
            MISNAMED, f"{MISNAMED}.{result_files.SET_ASIDE_SUFFIX}", [PI, CODEX]
        )
        assert f"**Result files:** {note}" in report

    def test_a_second_reaggregate_reads_the_same_two_models(self, tmp_path, capsys):
        plan, phase_dir = _live_code_review_dir(tmp_path)
        _run_cro_reaggregate(plan, phase_dir)
        capsys.readouterr()

        report = _run_cro_reaggregate(plan, phase_dir)
        out = capsys.readouterr().out

        assert sorted(_reported_results(report)) == sorted([PI, CODEX])
        assert "[RESULT_SET_ASIDE]" not in out
        assert len(read_status(str(phase_dir))["result_notes"]) == 1

    def test_the_review_plan_lane_sets_it_aside_too(self, tmp_path, capsys):
        plan, out_dir, phase_dir = _phase(tmp_path, {
            "claude-code_sonnet": _suggestions(["Alpha"]),
            "codex_gpt-6": _suggestions(["Bravo"]),
            "codex_gpt-6-old-panel": _suggestions(["Charlie"]),
        }, models_requested=["claude-code:sonnet", "codex:gpt-6"])

        _reaggregate(plan, out_dir, phase_dir)
        out = capsys.readouterr().out

        grouped = json.loads((phase_dir / "grouped.json").read_text(encoding="utf-8"))
        groups = grouped["groups"] if isinstance(grouped, dict) else grouped
        titles = {s["title"] for g in groups for s in g["suggestions"]}
        assert titles == {"Alpha", "Bravo"}
        assert "[RESULT_SET_ASIDE] codex_gpt-6-old-panel.json is not" in out
        assert (phase_dir / f"codex_gpt-6-old-panel.json.{result_files.SET_ASIDE_SUFFIX}").exists()

    def test_a_stray_that_is_not_a_result_list_is_left_where_it_is(self, tmp_path, capsys):
        plan, phase_dir = _cro_phase(tmp_path, {
            CODEX_STEM: _issues(["A"]),
        }, models_requested=[CODEX])
        (phase_dir / "notes.json").write_text(json.dumps({"k": "v"}), encoding="utf-8")

        report = _run_cro_reaggregate(plan, phase_dir)
        out = capsys.readouterr().out

        assert list(_reported_results(report)) == [CODEX]
        assert (phase_dir / "notes.json").exists()
        assert "notes.json is not a result file of any model this review ran" in out

    def test_a_fifo_under_a_result_name_is_not_opened(self, tmp_path, capsys):
        """A FIFO named `*.json` would hang the read that decides whether it is
        a result; it is named and left where it is instead."""
        import os as _os
        import threading
        plan, phase_dir = _cro_phase(tmp_path, {CODEX_STEM: _issues(["A"])},
                                     models_requested=[CODEX])
        fifo = phase_dir / "pi_stray.json"
        _os.mkfifo(fifo)
        spy = patch.object(cro, "resolve_result_files_by_model",
                           wraps=rob.resolve_result_files_by_model)
        with spy as resolved:
            # In a thread, so a regression that opens the FIFO fails here
            # instead of hanging the suite: a writer is opened to release it.
            worker = threading.Thread(
                target=_run_cro_reaggregate, args=(plan, phase_dir), daemon=True)
            worker.start()
            worker.join(timeout=30)
            if worker.is_alive():
                _os.close(_os.open(fifo, _os.O_WRONLY | _os.O_NONBLOCK))
                worker.join(timeout=30)
                pytest.fail("the reaggregate opened a FIFO named like a result")
        out = capsys.readouterr().out
        spy = resolved
        assert "[RESULT_NOT_READ] pi_stray.json" in out
        assert [p for p in spy.call_args.args[1] if p.endswith("pi_stray.json")] == []

    def test_a_dir_with_no_checkpoint_still_reads_every_file_by_stem(self, tmp_path, capsys):
        """No `.status.json`: nothing says which models ran, so nothing is a stray
        (the hand-copied dir keeps its old reading)."""
        plan, phase_dir = _cro_phase(tmp_path, {
            CODEX_STEM: _issues(["A"]),
            MISNAMED[: -len(".json")]: _issues(["B"]),
        })
        report = _run_cro_reaggregate(plan, phase_dir)
        capsys.readouterr()
        assert len(_reported_results(report)) == 2
        assert (phase_dir / MISNAMED).exists()


# ---------------------------------------------------------------------------
# Issue 2: what the host is told to say about a retry and about such a file
# ---------------------------------------------------------------------------

_REVIEW_RUNBOOKS = ("review-plan.md", "review-tasks.md", "review-code.md")


def _flat(name):
    text = (SKILL_ROOT / "instructions" / name).read_text(encoding="utf-8")
    return " ".join(text.split())


class TestTheRunbooksSayWhatARetryRanAndWhatAFileWas:
    """Live run 10: the host's `--force --models <both>` retry ran pi alone
    (`[SKIP] codex:… - already has results`), and it called the misnamed file
    "a leftover file from an earlier run" — words the plugin never printed."""

    @pytest.mark.parametrize("name", _REVIEW_RUNBOOKS)
    def test_the_retry_rule_names_the_skip_line_and_asks_which_model_reran(self, name):
        text = _flat(name)
        assert "`[SKIP] <model> - already has results`" in text
        assert "tell the user which model re-ran and which kept its result" in text

    @pytest.mark.parametrize("name", _REVIEW_RUNBOOKS)
    def test_the_report_step_relays_the_result_file_lines(self, name):
        text = _flat(name)
        assert "`[RESULT_FILE_MISNAMED] <model> wrote its result to <file> instead of <file>; …`" in text
        assert "`[RESULT_SET_ASIDE] <file> is not the result file of any model this review ran (…)`" in text
        assert "`**Result files:**`" in text
        assert "Never guess where such a file came from" in text

    def test_the_documented_skip_line_is_what_both_lanes_print(self):
        for path in ("code_review_orchestrator.py", "utils/review_orchestrator_base.py"):
            source = (SKILL_ROOT / path).read_text(encoding="utf-8")
            assert 'print(f"[SKIP] {display_name} - already has results (' in source, path

    def test_the_documented_marker_words_are_the_printed_ones(self):
        note = result_files.misnamed_result_note(PI, MISNAMED, f"{PI_STEM}.json")
        assert note.startswith(f"{PI} wrote its result to {MISNAMED} instead of {PI_STEM}.json; ")
        aside = result_files.set_aside_note(MISNAMED, f"{MISNAMED}.set-aside", [PI, CODEX])
        assert aside.startswith(f"{MISNAMED} is not the result file of any model this review ran (")
